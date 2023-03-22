# -*- coding: utf-8 -*-

"""Unet-baed HiFi-GAN Modules.

This code is based on https://github.com/jik876/hifi-gan.

"""

import logging

import numpy as np
import torch

from parallel_wavegan.layers import CausalConv1d, CausalConvTranspose1d
from parallel_wavegan.layers import HiFiGANResidualBlock as ResidualBlock
from parallel_wavegan.utils import read_hdf5


class UHiFiGANGenerator(torch.nn.Module):
    """UHiFiGAN generator module."""

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        downsample_scales=(8, 8, 2, 2),
        downsample_kernel_sizes=(16, 16, 4, 4),
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        dropout=0.3,
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_causal_conv=False,
        use_weight_norm=True,
    ):
        """Initialize Unet-based HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_causal_conv (bool): Whether to use causal structure.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.use_causal_conv = use_causal_conv
        self.downsamples = torch.nn.ModuleList()
        self.downsamples_mrf = torch.nn.ModuleList()
        self.upsamples = torch.nn.ModuleList()
        self.upsamples_mrf = torch.nn.ModuleList()

        if not use_causal_conv:
            self.input_conv = torch.nn.Sequential(
                torch.nn.Conv1d(
                    out_channels,
                    channels,
                    kernel_size=kernel_size,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Dropout(dropout),
            )
        else:
            self.input_conv = torch.nn.Sequential(
                CausalConv1d(
                    out_channels,
                    channels,
                    kernel_size=kernel_size,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Dropout(dropout),
            )

        for i in range(len(downsample_scales)):
            for j in range(len(resblock_kernel_sizes)):
                self.downsamples_mrf += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels,
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

            if not use_causal_conv:
                self.downsamples += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            channels,
                            channels * 2,
                            kernel_size=downsample_kernel_sizes[i],
                            stride=downsample_scales[i],
                            bias=bias,
                            padding=downsample_scales[i] // 2
                            + downsample_scales[i] % 2,
                        ),
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.Dropout(dropout),
                    )
                ]
            else:
                self.downsamples += [
                    torch.nn.Sequential(
                        CausalConv1d(
                            channels,
                            channels * 2,
                            kernel_size=downsample_kernel_sizes[i],
                            stride=downsample_scales[i],
                            bias=bias,
                            padding=downsample_scales[i] // 2
                            + downsample_scales[i] % 2,
                        ),
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.Dropout(dropout),
                    )
                ]

            channels = channels * 2

        if not use_causal_conv:
            self.hidden_conv = torch.nn.Conv1d(
                in_channels,
                channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            )
        else:
            self.hidden_conv = CausalConv1d(
                in_channels,
                channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            )

        for i in range(len(upsample_kernel_sizes)):
            # assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            if not use_causal_conv:
                self.upsamples += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.ConvTranspose1d(
                            channels * 2,
                            channels // 2,
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                            output_padding=upsample_scales[i] % 2,
                            bias=bias,
                        ),
                    )
                ]
            else:
                self.upsamples += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        CausalConvTranspose1d(
                            channels * 2,
                            channels // 2,
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            bias=bias,
                        ),
                    )
                ]

            # hidden_channel for MRF module
            for j in range(len(resblock_kernel_sizes)):
                self.upsamples_mrf += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // 2,
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

            channels = channels // 2

        if not use_causal_conv:
            self.output_conv = torch.nn.Sequential(
                # NOTE(kan-bayashi): follow official implementation but why
                #   using different slope parameter here? (0.1 vs. 0.01)
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    channels,
                    out_channels,
                    kernel_size,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.Tanh(),
            )
        else:
            self.output_conv = torch.nn.Sequential(
                # NOTE(kan-bayashi): follow official implementation but why
                #   using different slope parameter here? (0.1 vs. 0.01)
                torch.nn.LeakyReLU(),
                CausalConv1d(
                    channels,
                    out_channels,
                    kernel_size,
                    bias=bias,
                ),
                torch.nn.Tanh(),
            )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c=None, f0=None, excitation=None):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).
            f0 (Tensor): Input tensor (B, 1, T).
            excitation (Tensor): Input tensor (B, frame_len, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        residual_results = []
        hidden = self.input_conv(excitation)

        for i in range(len(self.downsamples)):
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                tc = self.downsamples_mrf[i * self.num_blocks + j](hidden)
                cs += tc
            hidden = cs / self.num_blocks
            hidden = self.downsamples[i](hidden)
            residual_results.append(hidden)

        residual_results.reverse()

        hidden_mel = self.hidden_conv(c)

        for i in range(len(self.upsamples)):
            hidden_mel = torch.cat((hidden_mel, residual_results[i]), dim=1)
            hidden_mel = self.upsamples[i](hidden_mel)

            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                tc = self.upsamples_mrf[i * self.num_blocks + j](hidden_mel)
                cs += tc
            hidden_mel = cs / self.num_blocks

        mel = self.output_conv(hidden_mel)

        return mel

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

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

    def inference(self, excitation=None, f0=None, c=None, normalize_before=False):
        """Perform inference.

        Args:
            excitation (Union[Tensor, ndarray]): Excitation tensor.
            f0 (Union[Tensor, ndarray]): F0 tensor.
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        if c is not None and not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if excitation is not None and not isinstance(excitation, torch.Tensor):
            excitation = torch.tensor(excitation, dtype=torch.float).to(
                next(self.parameters()).device
            )
        if f0 is not None and not isinstance(f0, torch.Tensor):
            f0 = torch.tensor(f0, dtype=torch.float).to(next(self.parameters()).device)

        c = self.forward(
            c.transpose(1, 0).unsqueeze(0),
            f0.unsqueeze(1).transpose(1, 0).unsqueeze(0),
            excitation.reshape(1, 1, -1),
        )
        return c.squeeze(0).transpose(1, 0)
