# -*- coding: utf-8 -*-

# Copyright 2020 MINH ANH (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tensorflow MelGAN modules complatible with pytorch."""

import tensorflow as tf

import numpy as np

from parallel_wavegan.layers.tf_layers import TFConvTranspose1d
from parallel_wavegan.layers.tf_layers import TFReflectionPad1d
from parallel_wavegan.layers.tf_layers import TFResidualStack


class TFMelGANGenerator(tf.keras.layers.Layer):
    """Tensorflow MelGAN generator module."""

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        kernel_size=7,
        channels=512,
        bias=True,
        upsample_scales=[8, 8, 2, 2],
        stack_kernel_size=3,
        stacks=3,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_final_nonlinear_activation=True,
        use_weight_norm=True,
        use_causal_conv=False,
    ):
        """Initialize TFMelGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (list): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_final_nonlinear_activation (torch.nn.Module): Activation function for the final layer.
            use_weight_norm (bool): No effect but keep it as is to be the same as pytorch version.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(TFMelGANGenerator, self).__init__()

        # check hyper parameters is valid
        assert not use_causal_conv, "Not supported yet."
        assert channels >= np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales)) == 0
        assert pad == "ReflectionPad1d", f"Not supported (pad={pad})."

        # add initial layer
        layers = []
        layers += [
            TFReflectionPad1d((kernel_size - 1) // 2),
            tf.keras.layers.Conv2D(
                filters=channels,
                kernel_size=(kernel_size, 1),
                padding="valid",
                use_bias=bias,
            ),
        ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [
                getattr(tf.keras.layers, nonlinear_activation)(
                    **nonlinear_activation_params
                ),
                TFConvTranspose1d(
                    channels=channels // (2 ** (i + 1)),
                    kernel_size=upsample_scale * 2,
                    stride=upsample_scale,
                    padding="same",
                ),
            ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    TFResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 1)),
                        dilation=stack_kernel_size ** j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        padding="same",
                    )
                ]

        # add final layer
        layers += [
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            ),
            TFReflectionPad1d((kernel_size - 1) // 2),
            tf.keras.layers.Conv2D(
                filters=out_channels, kernel_size=(kernel_size, 1), use_bias=bias
            ),
        ]
        if use_final_nonlinear_activation:
            layers += [tf.keras.layers.Activation("tanh")]

        self.melgan = tf.keras.models.Sequential(layers)

    # TODO(kan-bayashi): Fix hard coded dimension
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, None, 80], dtype=tf.float32)]
    )
    def call(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, T, in_channels).

        Returns:
            Tensor: Output tensor (B, T ** prod(upsample_scales), out_channels).

        """
        c = tf.expand_dims(c, 2)
        c = self.melgan(c)
        return c[:, :, 0, :]
