# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""StyleMelGAN Modules."""

import copy
import logging

import numpy as np
import torch

from parallel_wavegan.layers import PQMF
from parallel_wavegan.models import MelGANDiscriminator as BaseDiscriminator


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
        for idx, (ws, pqmf, disc) in enumerate(zip(self.window_sizes, self.pqmfs, self.discriminators)):
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
