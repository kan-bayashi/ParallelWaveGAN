# -*- coding: utf-8 -*-

"""Parallel WaveGAN Modules."""

import math

import torch

from parallel_wavegan.layers import Conv1d
from parallel_wavegan.layers import Conv1d1x1
from parallel_wavegan.layers import ResidualBlock
from parallel_wavegan.layers import upsample


class ParallelWaveGANGenerator(torch.nn.Module):
    """Parallel WaveGAN Generator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=30,
                 stacks=3,
                 residual_channels=64,
                 gate_channels=128,
                 skip_channels=64,
                 aux_channels=80,
                 dropout=1 - 0.95,
                 use_weight_norm=True,
                 upsample_conditional_features=True,
                 upsample_net="ConvInUpsampleNetwork",
                 upsample_params={"upsample_scales": [4, 4, 4, 4]},
                 ):
        """Initialize Parallel WaveGAN Generator module."""
        super(ParallelWaveGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size
        self.use_weight_norm = True

        # check the number of layers and stacks
        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        # define first convolution
        self.first_conv = self._apply_weight_norm(Conv1d1x1(in_channels, residual_channels))

        # define conv + upsampling network
        if upsample_conditional_features:
            self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
        else:
            self.upsample_net = None

        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            conv = ResidualBlock(
                Kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=dropout,
                use_weight_norm=use_weight_norm,
                bias=True,  # NOTE: magenda uses bias, but musyoku doesn't
                causal=False,
            )
            self.conv_layers += [conv]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList([
            torch.nn.ReLU(inplace=True),
            self._apply_weight_norm(Conv1d1x1(skip_channels, skip_channels)),
            torch.nn.ReLU(inplace=True),
            self._apply_weight_norm(Conv1d1x1(skip_channels, out_channels)),
        ])

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').

        Returns:
            Tensor: Output tensor (B, out_channels, T)

        """
        B, _, T = x.size()

        # perform upsampling
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == x.size(-1)

        # encode to hidden representation
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(_remove_weight_norm)

    def _apply_weight_norm(self, module):
        if self.use_weight_norm:
            return torch.nn.utils.weight_norm(module)
        else:
            return module

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size,
                                  dilation=lambda x: 2**x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        """Return receptive field size."""
        return self._get_receptive_field_size(self.layers, self.stacks, self.kernel_size)


class ParallelWaveGANDiscriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=10,
                 conv_channels=64,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 bias=True,
                 use_weight_norm=True,
                 ):
        """Initialize Parallel WaveGAN Discriminator module."""
        super(ParallelWaveGANDiscriminator, self).__init__()
        self.conv_layers = torch.nn.Module()
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
                conv_in_channels = in_channels
            else:
                dilation = i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [
                self._apply_weight_norm(Conv1d(
                    conv_in_channels, conv_channels,
                    kernel_size=kernel_size, padding=padding,
                    dilation=dilation, bias=bias)),
                getattr(torch.nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params)
            ]
            self.conv_layers += conv_layer
        last_conv_layer = self._apply_weight_norm(Conv1d(
            conv_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=bias))
        self.conv_layers += [last_conv_layer]

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        for f in self.conv_layers:
            x = f(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(_remove_weight_norm)

    def _apply_weight_norm(self, module):
        if self.use_weight_norm:
            return torch.nn.utils.weight_norm(module)
        else:
            return module
