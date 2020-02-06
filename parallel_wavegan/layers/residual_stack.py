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
                 nolinear_activation="LeakyReLU",
                 nolinear_activation_params={"negative_slope": 0.2},
                 padding_fn="ReflectionPad1d",
                 padding_params={},
                 use_causal_conv=False
                 ):
        """Initialize ResidualStack module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            nolinear_activation (str): Activation function module name.
            nolinear_activation_params (dict): Hyperparameters for activation function.
            padding_fn (str): Padding function module name before dilated convolution layer.
            padding_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(ResidualStack, self).__init__()

        assert not use_causal_conv, "Not supported yet."
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        padding = (kernel_size - 1) // 2 * dilation

        self.stack = torch.nn.Sequential(
            getattr(torch.nn, nolinear_activation)(**nolinear_activation_params),
            getattr(torch.nn, padding_fn)(padding, **padding_params),
            torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=bias),
            getattr(torch.nn, nolinear_activation)(**nolinear_activation_params),
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
