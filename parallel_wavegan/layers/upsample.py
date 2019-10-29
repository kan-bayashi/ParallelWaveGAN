# -*- coding: utf-8 -*-

"""Upsampling module.

This code is modified from https://github.com/r9y9/wavenet_vocoder.

"""

import numpy as np

import torch
from torch.nn import functional as F


class Stretch2d(torch.nn.Module):
    """Stretch2d module."""

    def __init__(self, x_scale, y_scale, mode="nearest"):
        """Initialize Stretch2d module."""
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        """Calculate forward propagation."""
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


class Conv2d(torch.nn.Conv2d):
    """Conv2d module with customized intialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d module."""
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        self.weight.data.fill_(1. / np.prod(self.kernel_size))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class UpsampleNetwork(torch.nn.Module):
    """Upsampling network module."""

    def __init__(self,
                 upsample_scales,
                 upsample_activation="none",
                 upsample_activation_params={},
                 mode="nearest",
                 freq_axis_kernel_size=1,
                 use_weight_norm=True,
                 ):
        """Initialize upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            upsample_activation (str): Activation function name.
            upsample_activation_params (dict): Arguments for specified activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
            use_weight_norm (bool): Whether to apply weight normalization.

        """
        super(UpsampleNetwork, self).__init__()
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_scales:
            # interpolatino layer
            stretch = Stretch2d(scale, 1, mode)
            self.up_layers += [stretch]

            # conv layer
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = (freq_axis_kernel_size, scale * 2 + 1)
            padding = (freq_axis_padding, scale)
            conv = Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            if use_weight_norm:
                conv = torch.nn.utils.weight_norm(conv)
            self.up_layers += [conv]

            # nonlinear
            if upsample_activation != "none":
                nonlinear = getattr(torch.nn, upsample_activation)(**upsample_activation_params)
                self.up_layers += [nonlinear]

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c : Input tensor (B, C, T).

        Returns:
            Tensor: Upsampled tensor (B, C, T'), where T' = T * prod(upsample_scales).

        """
        c = c.unsqueeze(1)  # (B, 1, C, T)
        for f in self.up_layers:
            c = f(c)
        c = c.squeeze(1)  # (B, C, T')

        return c


class ConvInUpsampleNetwork(torch.nn.Module):
    """Convolution + upsampling network module."""

    def __init__(self,
                 upsample_scales,
                 upsample_activation="none",
                 upsample_activation_params={},
                 mode="nearest",
                 freq_axis_kernel_size=1,
                 aux_channels=80,
                 aux_context_window=0,
                 use_weight_norm=True,
                 ):
        """Initialize convolution + upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            upsample_activation (str): Activation function name.
            upsample_activation_params (dict): Arguments for specified activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
            aux_context_window (int): Context window size of the pre-convolutional layer.
            use_weight_norm (bool): Whether to apply weight normalization.

        """
        super(ConvInUpsampleNetwork, self).__init__()
        # To capture wide-context information in conditional features
        kernel_size = 2 * aux_context_window + 1
        total_scale = np.prod(upsample_scales)
        self.indent = aux_context_window * total_scale
        conv_in = torch.nn.Conv1d(aux_channels, aux_channels, kernel_size=kernel_size, bias=False)
        if use_weight_norm:
            conv_in = torch.nn.utils.weight_norm(conv_in)
        self.conv_in = conv_in
        self.upsample = UpsampleNetwork(
            upsample_scales, upsample_activation, upsample_activation_params,
            mode, freq_axis_kernel_size, use_weight_norm)

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c : Input tensor (B, C, T).

        Returns:
            Tensor: Upsampled tensor (B, C, T'), where T' = T * prod(upsample_scales).

        """
        c = self.upsample(self.conv_in(c))

        # remove padded parts
        if self.indent > 0:
            c = c[:, :, self.indent:-self.indent]

        return c
