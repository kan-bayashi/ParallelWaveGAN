#!/usr/bin/env python3

# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Test code for StyleMelGAN modules."""

import logging

import pytest
import torch

from parallel_wavegan.losses import GeneratorAdversarialLoss
from parallel_wavegan.models import StyleMelGANDiscriminator


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def make_sytle_melgan_discriminator_args(**kwargs):
    defaults = dict(
        repeats=2,
        window_sizes=[512, 1024, 2048, 4096],
        pqmf_params=[
            [1, None, None, None],
            [2, 62, 0.26700, 9.0],
            [4, 62, 0.14200, 9.0],
            [8, 62, 0.07949, 9.0],
        ],
        discriminator_params={
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 16,
            "max_downsample_channels": 32,
            "bias": True,
            "downsample_scales": [4, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.2},
            "pad": "ReflectionPad1d",
            "pad_params": {},
        },
        use_weight_norm=True,
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "dict_d",
    [
        {"repeats": 1},
        {"repeats": 4},
    ],
)
def test_style_melgan_discriminator(dict_d):
    batch_size = 4
    batch_length = 2 ** 14
    args_d = make_sytle_melgan_discriminator_args(**dict_d)
    y = torch.randn(batch_size, 1, batch_length)
    model_d = StyleMelGANDiscriminator(**args_d)
    gen_adv_criterion = GeneratorAdversarialLoss()
    outs = model_d(y)
    gen_adv_criterion(outs)
