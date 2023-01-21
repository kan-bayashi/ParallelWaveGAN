# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""StyleMelGAN Modules."""

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F

from parallel_wavegan.layers import PQMF, TADEResBlock
from parallel_wavegan.models import MelGANDiscriminator as BaseDiscriminator
from parallel_wavegan.utils import read_hdf5


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
        noise_upsample_activation="LeakyReLU",
        noise_upsample_activation_params={"negative_slope": 0.2},
        upsample_scales=[2, 2, 2, 2, 2, 2, 2, 2, 1],
        upsample_mode="nearest",
        gated_function="softmax",
        use_weight_norm=True,
    ):
        """Initilize Style MelGAN generator.

        Args:
            in_channels (int): Number of input noise channels.
            aux_channels (int): Number of auxiliary input channels.
            channels (int): Number of channels for conv layer.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of conv layers.
            dilation (int): Dilation factor for conv layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            noise_upsample_scales (list): List of noise upsampling scales.
            noise_upsample_activation (str): Activation function module name for noise upsampling.
            noise_upsample_activation_params (dict): Hyperparameters for the above activation function.
            upsample_scales (list): List of upsampling scales.
            upsample_mode (str): Upsampling mode in TADE layer.
            gated_function (str): Gated function in TADEResBlock ("softmax" or "sigmoid").
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        self.in_channels = in_channels

        noise_upsample = []
        in_chs = in_channels
        for noise_upsample_scale in noise_upsample_scales:
            # NOTE(kan-bayashi): How should we design noise upsampling part?
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
                getattr(torch.nn, noise_upsample_activation)(
                    **noise_upsample_activation_params
                )
            ]
            in_chs = channels
        self.noise_upsample = torch.nn.Sequential(*noise_upsample)
        self.noise_upsample_factor = np.prod(noise_upsample_scales)

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
                    upsample_factor=upsample_scale,
                    upsample_mode=upsample_mode,
                    gated_function=gated_function,
                ),
            ]
            aux_chs = channels
        self.upsample_factor = np.prod(upsample_scales)

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

    def forward(self, c, z=None):
        """Calculate forward propagation.

        Args:
            c (Tensor): Auxiliary input tensor (B, channels, T).
            z (Tensor): Input noise tensor (B, in_channels, 1).

        Returns:
            Tensor: Output tensor (B, out_channels, T ** prod(upsample_scales)).

        """
        if z is None:
            z = torch.randn(c.size(0), self.in_channels, 1).to(
                device=c.device,
                dtype=c.dtype,
            )
        x = self.noise_upsample(z)
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

    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        if stats.endswith(".h5"):
            mean = read_hdf5(stats, "mean").reshape(-1)
            scale = read_hdf5(stats, "scale").reshape(-1)
        else:
            mean = np.load(stats)[0].reshape(-1)
            scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        logging.info("Successfully registered stats as buffer.")

    def inference(self, c, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = c.transpose(1, 0).unsqueeze(0)

        # prepare noise input
        noise_size = (
            1,
            self.in_channels,
            (c.size(2) - 1) // self.noise_upsample_factor + 1,
        )
        noise = torch.randn(*noise_size, dtype=torch.float).to(
            next(self.parameters()).device
        )
        x = self.noise_upsample(noise)

        # NOTE(kan-bayashi): To remove pop noise at the end of audio, perform padding
        #    for feature sequence and after generation cut the generated audio. This
        #    requires additional computation but it can prevent pop noise.
        total_length = c.size(2) * self.upsample_factor
        c = F.pad(c, (0, x.size(2) - c.size(2)), "replicate")

        # This version causes pop noise.
        # x = x[:, :, :c.size(2)]

        for block in self.blocks:
            x, c = block(x, c)
        x = self.output_conv(x)[..., :total_length]

        return x.squeeze(0).transpose(1, 0)


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

        # window size check
        assert len(window_sizes) == len(pqmf_params)
        sizes = [ws // p[0] for ws, p in zip(window_sizes, pqmf_params)]
        assert len(window_sizes) == sum([sizes[0] == size for size in sizes])

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


class DiscreteSymbolStyleMelGANGenerator(torch.nn.Module):
    """Discrete Symbol Style MelGAN generator module."""

    def __init__(
        self,
        in_channels=128,
        aux_channels=128,
        channels=64,
        out_channels=1,
        num_embs=100,
        num_spk_embs=128,
        spk_emb_dim=128,
        concat_spk_emb=False,
        kernel_size=9,
        dilation=2,
        bias=True,
        noise_upsample_scales=[11, 2, 2, 2],
        noise_upsample_activation="LeakyReLU",
        noise_upsample_activation_params={"negative_slope": 0.2},
        upsample_scales=[2, 2, 2, 2, 2, 2, 2, 2, 1],
        upsample_mode="nearest",
        gated_function="softmax",
        use_weight_norm=True,
    ):
        """Initilize Style MelGAN generator.

        Args:
            in_channels (int): Number of input noise channels.
            aux_channels (int): Number of auxiliary input channels.
            channels (int): Number of channels for conv layer.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of conv layers.
            dilation (int): Dilation factor for conv layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            noise_upsample_scales (list): List of noise upsampling scales.
            noise_upsample_activation (str): Activation function module name for noise upsampling.
            noise_upsample_activation_params (dict): Hyperparameters for the above activation function.
            upsample_scales (list): List of upsampling scales.
            upsample_mode (str): Upsampling mode in TADE layer.
            gated_function (str): Gated function in TADEResBlock ("softmax" or "sigmoid").
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        self.in_channels = in_channels

        # define id embedding
        self.emb = torch.nn.Embedding(
            num_embeddings=num_embs, embedding_dim=aux_channels
        )
        self.spk_emb = torch.nn.Embedding(
            num_embeddings=num_spk_embs, embedding_dim=spk_emb_dim
        )
        self.concat_spk_emb = concat_spk_emb
        if not concat_spk_emb:
            assert aux_channels == spk_emb_dim
        else:
            aux_channels = aux_channels + spk_emb_dim

        noise_upsample = []
        in_chs = in_channels
        for noise_upsample_scale in noise_upsample_scales:
            # NOTE(kan-bayashi): How should we design noise upsampling part?
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
                getattr(torch.nn, noise_upsample_activation)(
                    **noise_upsample_activation_params
                )
            ]
            in_chs = channels
        self.noise_upsample = torch.nn.Sequential(*noise_upsample)
        self.noise_upsample_factor = np.prod(noise_upsample_scales)

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
                    upsample_factor=upsample_scale,
                    upsample_mode=upsample_mode,
                    gated_function=gated_function,
                ),
            ]
            aux_chs = channels
        self.upsample_factor = np.prod(upsample_scales)

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

    def forward(self, c, z=None):
        """Calculate forward propagation.

        Args:
            c (Tensor): Auxiliary input tensor (B, channels, T).
            z (Tensor): Input noise tensor (B, in_channels, 1).

        Returns:
            Tensor: Output tensor (B, out_channels, T ** prod(upsample_scales)).

        """
        # convert idx to embedding
        assert c.size(1) == 2
        c_idx, g_idx = c.long().split(1, dim=1)
        c = self.emb(c_idx.squeeze(1)).transpose(1, 2)  # (B, C, T)
        g = self.spk_emb(g_idx[:, 0, 0])

        # integrate global embedding
        if not self.concat_spk_emb:
            c = c + g.unsqueeze(2)
        else:
            g = g.unsqueeze(1).expand(-1, c.size(1), -1)
            c = torch.cat([c, g], dim=-1)

        if z is None:
            z = torch.randn(c.size(0), self.in_channels, 1).to(
                device=c.device,
                dtype=c.dtype,
            )
        x = self.noise_upsample(z)
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

    def inference(self, c, g=None, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, 2).

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        assert not normalize_before, "No statistics are used."
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.long).to(next(self.parameters()).device)
        if g is not None:
            c = c[:, 0:1]
            c = torch.cat([c, c.new_zeros(*c.size()).fill_(g).to(c.device)], dim=1)
        c = c.transpose(1, 0).unsqueeze(0)

        # convert idx to embedding
        assert c.size(1) == 2
        c_idx, g_idx = c.long().split(1, dim=1)
        c = self.emb(c_idx.squeeze(1)).transpose(1, 2)  # (B, C, T)
        g = self.spk_emb(g_idx[:, 0, 0])

        # integrate global embedding
        if not self.concat_spk_emb:
            c = c + g.unsqueeze(2)
        else:
            g = g.unsqueeze(1).expand(-1, c.size(1), -1)
            c = torch.cat([c, g], dim=-1)

        noise_size = (
            1,
            self.in_channels,
            (c.size(2) - 1) // self.noise_upsample_factor + 1,
        )
        noise = torch.randn(*noise_size, dtype=torch.float).to(
            next(self.parameters()).device
        )
        noise_up = self.noise_upsample(noise)
        x = noise_up[:, :, : c.size(2)]
        for block in self.blocks:
            x, c = block(x, c)
        x = self.output_conv(x)

        return x.squeeze(0).transpose(1, 0)
