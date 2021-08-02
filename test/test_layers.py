#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

import logging

import numpy as np
import pytest
import torch

from parallel_wavegan.layers import CausalConv1d
from parallel_wavegan.layers import CausalConvTranspose1d
from parallel_wavegan.layers import Conv1d
from parallel_wavegan.layers import Conv1d1x1
from parallel_wavegan.layers import Conv2d
from parallel_wavegan.layers import ConvInUpsampleNetwork
from parallel_wavegan.layers import PQMF
from parallel_wavegan.layers import UpsampleNetwork

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def test_conv_initialization():
    conv = Conv1d(10, 10, 3, bias=True)
    np.testing.assert_array_equal(
        conv.bias.data.numpy(), np.zeros_like(conv.bias.data.numpy())
    )
    conv1x1 = Conv1d1x1(10, 10, bias=True)
    np.testing.assert_array_equal(
        conv1x1.bias.data.numpy(), np.zeros_like(conv1x1.bias.data.numpy())
    )
    kernel_size = (10, 10)
    conv2d = Conv2d(10, 10, kernel_size, bias=True)
    np.testing.assert_array_equal(
        conv2d.weight.data.numpy(),
        np.ones_like(conv2d.weight.data.numpy()) / np.prod(kernel_size),
    )
    np.testing.assert_array_equal(
        conv2d.bias.data.numpy(), np.zeros_like(conv2d.bias.data.numpy())
    )
    kernel_size = (1, 10)
    conv2d = Conv2d(10, 10, kernel_size, bias=True)
    np.testing.assert_array_equal(
        conv2d.weight.data.numpy(),
        np.ones_like(conv2d.weight.data.numpy()) / np.prod(kernel_size),
    )
    np.testing.assert_array_equal(
        conv2d.bias.data.numpy(), np.zeros_like(conv2d.bias.data.numpy())
    )


@pytest.mark.parametrize(
    "use_causal_conv",
    [
        (False),
        (True),
    ],
)
def test_upsample(use_causal_conv):
    length = 10
    scales = [4, 4]
    x = torch.randn(1, 10, length)
    upsample = UpsampleNetwork(scales)
    y = upsample(x)
    assert x.size(-1) * np.prod(scales) == y.size(-1)

    for aux_context_window in [0, 1, 2, 3]:
        conv_upsample = ConvInUpsampleNetwork(
            scales,
            aux_channels=x.size(1),
            aux_context_window=aux_context_window,
            use_causal_conv=use_causal_conv,
        )
        y = conv_upsample(x)
        assert (x.size(-1) - 2 * aux_context_window) * np.prod(scales) == y.size(-1)


@torch.no_grad()
@pytest.mark.parametrize(
    "kernel_size, dilation, pad, pad_params",
    [
        (3, 1, "ConstantPad1d", {"value": 0.0}),
        (3, 3, "ConstantPad1d", {"value": 0.0}),
        (2, 1, "ConstantPad1d", {"value": 0.0}),
        (2, 3, "ConstantPad1d", {"value": 0.0}),
        (5, 1, "ConstantPad1d", {"value": 0.0}),
        (5, 3, "ConstantPad1d", {"value": 0.0}),
        (3, 3, "ReflectionPad1d", {}),
        (2, 1, "ReflectionPad1d", {}),
        (2, 3, "ReflectionPad1d", {}),
        (5, 1, "ReflectionPad1d", {}),
        (5, 3, "ReflectionPad1d", {}),
    ],
)
def test_causal_conv(kernel_size, dilation, pad, pad_params):
    x = torch.randn(1, 1, 32)
    conv = CausalConv1d(1, 1, kernel_size, dilation, pad=pad, pad_params=pad_params)
    y1 = conv(x)
    x[:, :, 16:] += torch.randn(1, 1, 16)
    y2 = conv(x)
    assert x.size(2) == y1.size(2)
    np.testing.assert_array_equal(
        y1[:, :, :16].cpu().numpy(),
        y2[:, :, :16].cpu().numpy(),
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "kernel_size, stride",
    [
        (4, 2),
        (6, 3),
        (10, 5),
    ],
)
def test_causal_conv_transpose(kernel_size, stride):
    deconv = CausalConvTranspose1d(1, 1, kernel_size, stride)
    x = torch.randn(1, 1, 32)
    y1 = deconv(x)
    x[:, :, 19:] += torch.randn(1, 1, 32 - 19)
    y2 = deconv(x)
    assert x.size(2) * stride == y1.size(2)
    np.testing.assert_array_equal(
        y1[:, :, : 19 * stride].cpu().numpy(),
        y2[:, :, : 19 * stride].cpu().numpy(),
    )


@pytest.mark.parametrize(
    "subbands",
    [
        (3),
        (4),
    ],
)
def test_pqmf(subbands):
    pqmf = PQMF(subbands)
    x = torch.randn(1, 1, subbands * 32)
    y = pqmf.analysis(x)
    assert y.shape[2] * subbands == x.shape[2]
    x_hat = pqmf.synthesis(y)
    assert x.shape[2] == x_hat.shape[2]
