#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

import logging

import numpy as np
import pytest
import torch

from parallel_wavegan.layers import Conv1d
from parallel_wavegan.layers import Conv1d1x1
from parallel_wavegan.layers import Conv2d
from parallel_wavegan.layers import ConvInUpsampleNetwork
from parallel_wavegan.layers import UpsampleNetwork

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")


def test_conv_initialization():
    conv = Conv1d(10, 10, 3, bias=True)
    np.testing.assert_array_equal(conv.bias.data.numpy(),
                                  np.zeros_like(conv.bias.data.numpy()))
    conv1x1 = Conv1d1x1(10, 10, bias=True)
    np.testing.assert_array_equal(conv1x1.bias.data.numpy(),
                                  np.zeros_like(conv1x1.bias.data.numpy()))
    kernel_size = (10, 10)
    conv2d = Conv2d(10, 10, kernel_size, bias=True)
    np.testing.assert_array_equal(conv2d.weight.data.numpy(),
                                  np.ones_like(conv2d.weight.data.numpy()) / np.prod(kernel_size))
    np.testing.assert_array_equal(conv2d.bias.data.numpy(),
                                  np.zeros_like(conv2d.bias.data.numpy()))
    kernel_size = (1, 10)
    conv2d = Conv2d(10, 10, kernel_size, bias=True)
    np.testing.assert_array_equal(conv2d.weight.data.numpy(),
                                  np.ones_like(conv2d.weight.data.numpy()) / np.prod(kernel_size))
    np.testing.assert_array_equal(conv2d.bias.data.numpy(),
                                  np.zeros_like(conv2d.bias.data.numpy()))


@pytest.mark.parametrize(
    "use_causal_conv", [
        (False),
        (True),
    ])
def test_upsample(use_causal_conv):
    length = 10
    scales = [4, 4]
    x = torch.randn(1, 10, length)
    upsample = UpsampleNetwork(scales)
    y = upsample(x)
    assert x.size(-1) * np.prod(scales) == y.size(-1)

    for aux_context_window in [0, 1, 2, 3]:
        conv_upsample = ConvInUpsampleNetwork(scales,
                                              aux_channels=x.size(1),
                                              aux_context_window=aux_context_window,
                                              use_causal_conv=use_causal_conv)
        y = conv_upsample(x)
        assert (x.size(-1) - 2 * aux_context_window) * np.prod(scales) == y.size(-1)
