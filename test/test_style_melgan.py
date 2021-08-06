#!/usr/bin/env python3

# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Test code for StyleMelGAN modules."""

import logging

import numpy as np
import pytest
import torch

from parallel_wavegan.losses import DiscriminatorAdversarialLoss
from parallel_wavegan.losses import GeneratorAdversarialLoss
from parallel_wavegan.losses import MultiResolutionSTFTLoss
from parallel_wavegan.models import StyleMelGANDiscriminator
from parallel_wavegan.models import StyleMelGANGenerator

from test_parallel_wavegan import make_mutli_reso_stft_loss_args


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def make_style_melgan_generator_args(**kwargs):
    defaults = dict(
        in_channels=128,
        aux_channels=80,
        channels=64,
        out_channels=1,
        kernel_size=9,
        dilation=2,
        bias=True,
        noise_upsample_scales=[11, 2, 2, 2],
        noise_upsample_activation="LeakyReLU",
        noise_upsample_activation_params={"negative_slope": 0.2},
        upsample_scales=[2, 2, 2, 2, 2, 2, 2, 2, 1],
        upsample_mode="nearest",
        gated_function="softmax",
        use_weight_norm=True,
    )
    defaults.update(kwargs)
    return defaults


def make_style_melgan_discriminator_args(**kwargs):
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
    args_d = make_style_melgan_discriminator_args(**dict_d)
    y = torch.randn(batch_size, 1, batch_length)
    model_d = StyleMelGANDiscriminator(**args_d)
    gen_adv_criterion = GeneratorAdversarialLoss()
    outs = model_d(y)
    gen_adv_criterion(outs)


@pytest.mark.parametrize(
    "dict_g",
    [
        {},
        {"noise_upsample_scales": [4, 4, 4]},
    ],
)
def test_style_melgan_generator(dict_g):
    args_g = make_style_melgan_generator_args(**dict_g)
    batch_size = 4
    batch_length = np.prod(args_g["noise_upsample_scales"]) * np.prod(
        args_g["upsample_scales"]
    )
    z = torch.randn(batch_size, args_g["in_channels"], 1)
    c = torch.randn(
        batch_size,
        args_g["aux_channels"],
        batch_length // np.prod(args_g["upsample_scales"]),
    )
    model_g = StyleMelGANGenerator(**args_g)
    model_g(c, z)

    # inference
    c = torch.randn(
        512,
        args_g["aux_channels"],
    )
    y = model_g.inference(c)
    print(y.shape)


@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss, loss_type",
    [
        ({}, {}, {}, "mse"),
        ({}, {}, {}, "hinge"),
        ({"noise_upsample_scales": [4, 4, 4]}, {}, {}, "mse"),
        ({"gated_function": "sigmoid"}, {}, {}, "mse"),
    ],
)
def test_style_melgan_trainable(dict_g, dict_d, dict_loss, loss_type):
    # setup
    args_g = make_style_melgan_generator_args(**dict_g)
    args_d = make_style_melgan_discriminator_args(**dict_d)
    args_loss = make_mutli_reso_stft_loss_args(**dict_loss)
    batch_size = 4
    batch_length = np.prod(args_g["noise_upsample_scales"]) * np.prod(
        args_g["upsample_scales"]
    )
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(
        batch_size,
        args_g["aux_channels"],
        batch_length // np.prod(args_g["upsample_scales"]),
    )
    model_g = StyleMelGANGenerator(**args_g)
    model_d = StyleMelGANDiscriminator(**args_d)
    aux_criterion = MultiResolutionSTFTLoss(**args_loss)
    gen_adv_criterion = GeneratorAdversarialLoss(loss_type=loss_type)
    dis_adv_criterion = DiscriminatorAdversarialLoss(loss_type=loss_type)
    optimizer_g = torch.optim.Adam(model_g.parameters())
    optimizer_d = torch.optim.Adam(model_d.parameters())

    # check generator trainable
    y_hat = model_g(c)
    p_hat = model_d(y_hat)
    adv_loss = gen_adv_criterion(p_hat)
    sc_loss, mag_loss = aux_criterion(y_hat, y)
    aux_loss = sc_loss + mag_loss
    loss_g = adv_loss + aux_loss
    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    # check discriminator trainable
    p = model_d(y)
    p_hat = model_d(y_hat.detach())
    real_loss, fake_loss = dis_adv_criterion(p_hat, p)
    loss_d = real_loss + fake_loss
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()
