# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Causal convolusion layer modules."""


import torch


class CausalConv1d(torch.nn.Module):
    """CausalConv1d module with customized initialization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        bias=True,
        pad="ConstantPad1d",
        pad_params={"value": 0.0},
    ):
        """Initialize CausalConv1d module."""
        super(CausalConv1d, self).__init__()
        self.pad = getattr(torch.nn, pad)((kernel_size - 1) * dilation, **pad_params)
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=bias
        )

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        return self.conv(self.pad(x))[:, :, : x.size(2)]


class CausalConvTranspose1d(torch.nn.Module):
    """CausalConvTranspose1d module with customized initialization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=True,
        pad="ReplicationPad1d",
        pad_params={},
    ):
        """Initialize CausalConvTranspose1d module."""
        super(CausalConvTranspose1d, self).__init__()
        # NOTE (yoneyama): This padding is to match the number of inputs
        #   used to calculate the first output sample with the others.
        self.pad = getattr(torch.nn, pad)((1, 0), **pad_params)
        self.deconv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, bias=bias
        )
        self.stride = stride

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T_in).

        Returns:
            Tensor: Output tensor (B, out_channels, T_out).

        """
        return self.deconv(self.pad(x))[:, :, self.stride : -self.stride]
