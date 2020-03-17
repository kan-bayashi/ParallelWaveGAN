# -*- coding: utf-8 -*-

# Copyright 2020 MINH ANH (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tensorflow Layer modules complatible with pytorch."""

import tensorflow as tf


class TFReflectionPad1d(tf.keras.layers.Layer):
    """Tensorflow ReflectionPad1d module."""

    def __init__(self, padding_size):
        """Initialize TFReflectionPad1d module."""
        super(TFReflectionPad1d, self).__init__()
        self.padding_size = padding_size

    @tf.function
    def call(self, x):
        """Calculate forward propagation."""
        return tf.pad(x, [[0, 0], [self.padding_size, self.padding_size], [0, 0], [0, 0]], "REFLECT")


class TFConvTranspose1d(tf.keras.layers.Layer):
    """Tensorflow ConvTranspose1d module."""

    def __init__(self, filters, kernel_size, strides, padding):
        """Initialize TFConvTranspose1d( module."""
        super(TFConvTranspose1d, self).__init__()
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding=padding,
        )

    @tf.function
    def call(self, x):
        """Calculate forward propagation."""
        x = self.conv1d_transpose(x)
        return x


class TFResidualStack(tf.keras.layers.Layer):
    """Tensorflow ResidualStack module."""

    def __init__(self,
                 kernel_size,
                 channels,
                 dilation,
                 bias,
                 nonlinear_activation,
                 nonlinear_activation_params,
                 padding,
                 ):
        """Initialize TFResidualStack module."""
        super(TFResidualStack, self).__init__()
        self.block = [
            getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params),
            TFReflectionPad1d(dilation),
            tf.keras.layers.Conv2D(
                filters=channels,
                kernel_size=(kernel_size, 1),
                dilation_rate=(dilation, 1),
                use_bias=bias,
                padding="valid",
            ),
            getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params),
            tf.keras.layers.Conv2D(filters=channels, kernel_size=1, use_bias=bias)
        ]
        self.shortcut = tf.keras.layers.Conv2D(filters=channels, kernel_size=1, use_bias=bias)

    @tf.function
    def call(self, x):
        """Calculate forward propagation."""
        _x = tf.identity(x)
        for i, layer in enumerate(self.block):
            _x = layer(_x)
        shortcut = self.shortcut(x)
        return shortcut + _x
