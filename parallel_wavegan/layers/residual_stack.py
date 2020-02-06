# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Residual stack module in MelGAN."""

import torch


class ResidualStack(torch.nn.Module):
    """Residual stack module introduced in MelGAN."""

    def __init__(self,
                 kernel_size=3,
                 channels=32,
                 dilation=1,
                 bias=True,
                 padding_fn=torch.nn.ReflectionPad1d,
                 activation_fn=torch.nn.LeakyReLU,
                 activation_params={"negative_slope": 0.2},
                 ):
        """Initialize ResidualStack module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            padding_fn (torch.nn.Module): Padding function before dilated convolution layer.
            activation_fn (torch.nn.Module): Activation function.
            activation_params (dict): Hyperparameters for activation function.

        """
        super(ResidualStack, self).__init__()

        self.stack = torch.nn.Sequential(
            activation_fn(**activation_params),
            padding_fn(dilation),
            torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=bias),
            activation_fn(**activation_params),
            torch.nn.Conv1d(channels, channels, 1, bias=bias),
        )

        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, chennels, T).

        """
        return self.stack(c) + self.skip_layer(c)
