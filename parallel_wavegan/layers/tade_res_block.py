# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""StyleMelGAN's TADEResBlock Modules."""

import torch
import torch.nn.functional as F


class TADELayer(torch.nn.Module):
    """TADE Layer module."""

    def __init__(
        self,
        in_channels=64,
        aux_channels=80,
        kernel_size=9,
        bias=True,
        in_upsample_factor=2,
        in_upsample_mode="nearest",
        aux_upsample_factor=2,
        aux_upsample_mode="nearest",
    ):
        """Initilize TADE layer."""
        super().__init__()
        self.norm = torch.nn.InstanceNorm1d(in_channels)
        self.aux_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                aux_channels,
                in_channels,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            # NOTE(kan-bayashi): Use non-linear activation?
        )
        self.gated_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels,
                in_channels * 2,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            # NOTE(kan-bayashi): Use non-linear activation?
        )
        self.in_upsample = torch.nn.Upsample(
            scale_factor=in_upsample_factor, mode=in_upsample_mode
        )
        self.aux_upsample = torch.nn.Upsample(
            scale_factor=aux_upsample_factor, mode=aux_upsample_mode
        )

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Auxiliary input tensor (B, aux_channels, T').

        Returns:
            Tensor: Output tensor (B, in_channels, T * in_upsample_factor).
            Tensor: Upsampled aux tensor (B, in_channels, T * aux_upsample_factor).

        """
        x = self.norm(x)
        c = self.aux_upsample(c)
        c = self.aux_conv(c)
        cg = self.gated_conv(c)
        cg1, cg2 = cg.split(cg.size(1) // 2, dim=1)
        # NOTE(kan-bayashi): Use upsample for noise input here?
        return cg1 * self.in_upsample(x) + cg2, c


class TADEResBlock(torch.nn.Module):
    """TADEResBlock module."""

    def __init__(
        self,
        in_channels=64,
        aux_channels=80,
        kernel_size=9,
        dilation=2,
        bias=True,
        in_upsample_factor=2,
        in_upsample_mode="nearest",
        aux_upsample_factor=2,
        aux_upsample_mode="nearest",
    ):
        """Initialize TADEResBlock module."""
        super().__init__()
        self.tade1 = TADELayer(
            in_channels=in_channels,
            aux_channels=aux_channels,
            kernel_size=kernel_size,
            bias=bias,
            # NOTE(kan-bayashi): Use upsample in the first TADE layer?
            in_upsample_factor=1,
            in_upsample_mode=in_upsample_mode,
            # NOTE(kan-bayashi): Use upsample in the first TADE layer?
            aux_upsample_factor=1,
            aux_upsample_mode=aux_upsample_mode,
        )
        self.gated_conv1 = torch.nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )
        self.tade2 = TADELayer(
            in_channels=in_channels,
            aux_channels=in_channels,
            kernel_size=kernel_size,
            bias=bias,
            in_upsample_factor=in_upsample_factor,
            in_upsample_mode=in_upsample_mode,
            aux_upsample_factor=aux_upsample_factor,
            aux_upsample_mode=aux_upsample_mode,
        )
        self.gated_conv2 = torch.nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias=bias,
            dilation=dilation,
            padding=(kernel_size - 1) // 2 * dilation,
        )
        self.upsample = torch.nn.Upsample(
            scale_factor=in_upsample_factor, mode=in_upsample_mode
        )

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Auxiliary input tensor (B, aux_channels, T').

        Returns:
            Tensor: Output tensor (B, in_channels, T * in_upsample_factor).

        """
        residual = x
        x, c = self.tade1(x, c)
        x = self.gated_conv1(x)
        xa, xb = x.split(x.size(1) // 2, dim=1)
        x = torch.softmax(xa, dim=1) * torch.tanh(xb)
        x, c = self.tade2(x, c)
        x = self.gated_conv2(x)
        xa, xb = x.split(x.size(1) // 2, dim=1)
        x = torch.softmax(xa, dim=1) * torch.tanh(xb)
        return self.upsample(residual) + x, c
