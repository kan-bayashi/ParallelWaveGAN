# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""MelGAN Modules."""

import logging

import numpy as np
import torch

from parallel_wavegan.layers import ResidualStack


class MelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 kernel_size=7,
                 channels=512,
                 bias=True,
                 upsample_scales=[8, 8, 2, 2],
                 stack_kernel_size=3,
                 stacks=3,
                 padding_fn=torch.nn.ReflectionPad1d,
                 activation_fn=torch.nn.LeakyReLU,
                 activation_params={"negative_slope": 0.2},
                 final_activation_fn=torch.nn.Tanh,
                 use_weight_norm=True,
                 ):
        """Initialize MelGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (list): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual stack.
            stacks (int): Number of stacks in a single residual stack.
            padding_fn (torch.nn.Module): Padding function before dilated convolution layer.
            activation_fn (torch.nn.Module): Activation function.
            activation_params (dict): Hyperparameters for activation function.
            final_activation_fn (torch.nn.Module): Activation function for the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        # check hyper parameters is valid
        assert channels > np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales))

        # add initial layer
        layers = []
        layers += [
            padding_fn((kernel_size - 1 // 2)),
            torch.nn.Conv1d(in_channels, channels, bias=bias),
        ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [
                activation_fn(**activation_params),
                torch.nn.ConvTranspose1d(
                    channels // (2 ** (i + 1)),
                    channels // (2 ** (i + 2)),
                    upsample_scale * 2,
                    stried=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
                )
            ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 2)),
                        dilation=stack_kernel_size ** j,
                        bias=bias,
                        padding_fn=padding_fn,
                        activation_fn=activation_fn,
                        activation_params=activation_params,
                    )
                ]

        # add final layer
        layers += [
            activation_fn(**activation_params),
            padding_fn((kernel_size - 1 // 2)),
            torch.nn.Conv1d(channels // (2 ** (i + 2)), 1, kernel_size, bias=bias),
        ]
        if final_activation_fn is not None:
            layers += [final_activation_fn()]

        self.melgan = torch.nn.Sequential(*layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, 1, T ** prod(upsample_scales)).

        """
        self.melgan(c)

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
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)
