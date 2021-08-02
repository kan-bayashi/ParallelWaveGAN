# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""StyleMelGAN Modules."""

import copy
import logging

import numpy as np
import torch

from parallel_wavegan.layers import PQMF
from parallel_wavegan.layers import TADEResBlock
from parallel_wavegan.models import MelGANDiscriminator as BaseDiscriminator


class StyleMelGANGenerator(torch.nn.Module):
    """Style MelGAN generator module."""

    def __init__(
        self,
        in_channels=128,
        aux_channels=80,
        channels=64,
        out_channels=1,
        kernel_size=9,
        dilation=2,
        bias=True,
        noise_upsample_scales=[11, 2, 2, 2],
        upsample_scales=[2, 2, 2, 2, 2, 2, 2, 2, 1],
        upsample_mode="nearest",
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        use_weight_norm=True,
    ):
        """Initilize Style MelGAN generator."""
        super().__init__()

        noise_upsample = []
        in_chs = in_channels
        for noise_upsample_scale in noise_upsample_scales:
            noise_upsample += [
                torch.nn.ConvTranspose1d(
                    in_chs,
                    channels,
                    noise_upsample_scale * 2,
                    stride=noise_upsample_scale,
                    padding=noise_upsample_scale // 2 + noise_upsample_scale % 2,
                    output_padding=noise_upsample_scale % 2,
                    bias=bias,
                )
            ]
            noise_upsample += [
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
            ]
            in_chs = channels
        self.noise_upsample = torch.nn.Sequential(*noise_upsample)

        self.blocks = torch.nn.ModuleList()
        aux_chs = aux_channels
        for upsample_scale in upsample_scales:
            self.blocks += [
                TADEResBlock(
                    in_channels=channels,
                    aux_channels=aux_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    bias=bias,
                    in_upsample_factor=upsample_scale,
                    in_upsample_mode=upsample_mode,
                    aux_upsample_factor=upsample_scale,
                    aux_upsample_mode=upsample_mode,
                ),
            ]
            aux_chs = channels

        self.output_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                channels,
                out_channels,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Tanh(),
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise tensor (B, 128, 1).
            c (Tensor): Auxiliary input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T ** prod(upsample_scales)).

        """
        x = self.noise_upsample(x)
        for block in self.blocks:
            x, c = block(x, c)
        x = self.output_conv(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters."""

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)


class StyleMelGANDiscriminator(torch.nn.Module):
    """Style MelGAN disciminator module."""

    def __init__(
        self,
        repeats=2,
        window_sizes=[512, 1024, 2048, 4096],
        pqmf_params=[
            [1, None, None, None],
            [2, 62, 0.26700, 9.0],
            [4, 62, 0.14200, 9.0],
            [8, 62, 0.07949, 9.0],
        ],
        discriminator_params={
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 16,
            "max_downsample_channels": 512,
            "bias": True,
            "downsample_scales": [4, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.2},
            "pad": "ReflectionPad1d",
            "pad_params": {},
        },
        use_weight_norm=True,
    ):
        """Initilize Style MelGAN discriminator.

        Args:
            repeats (int): Number of repititons to apply RWD.
            window_sizes (list): List of random window sizes.
            pqmf_params (list): List of list of Parameters for PQMF modules
            discriminator_params (dict): Parameters for base discriminator module.
            use_weight_nom (bool): Whether to apply weight normalization.

        """
        super().__init__()
        self.repeats = repeats
        self.window_sizes = window_sizes
        self.pqmfs = torch.nn.ModuleList()
        self.discriminators = torch.nn.ModuleList()
        for pqmf_param in pqmf_params:
            d_params = copy.deepcopy(discriminator_params)
            d_params["in_channels"] = pqmf_param[0]
            if pqmf_param[0] == 1:
                self.pqmfs += [torch.nn.Identity()]
            else:
                self.pqmfs += [PQMF(*pqmf_param)]
            self.discriminators += [BaseDiscriminator(**d_params)]

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, 1, T).

        Returns:
            List: List of discriminator outputs, #items in the list will be
                equal to repeats * #discriminators.

        """
        outs = []
        for _ in range(self.repeats):
            outs += self._forward(x)

        return outs

    def _forward(self, x):
        outs = []
        for idx, (ws, pqmf, disc) in enumerate(
            zip(self.window_sizes, self.pqmfs, self.discriminators)
        ):
            # NOTE(kan-bayashi): Is it ok to apply different window for real and fake samples?
            start_idx = np.random.randint(x.size(-1) - ws)
            x_ = x[:, :, start_idx : start_idx + ws]
            if idx == 0:
                x_ = pqmf(x_)
            else:
                x_ = pqmf.analysis(x_)
            outs += [disc(x_)]
        return outs

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters."""

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)
