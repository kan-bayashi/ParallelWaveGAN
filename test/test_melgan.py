#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

import logging

import numpy as np
import pytest
import torch

from parallel_wavegan.losses import DiscriminatorAdversarialLoss
from parallel_wavegan.losses import FeatureMatchLoss
from parallel_wavegan.losses import GeneratorAdversarialLoss
from parallel_wavegan.losses import MultiResolutionSTFTLoss
from parallel_wavegan.models import MelGANGenerator
from parallel_wavegan.models import MelGANMultiScaleDiscriminator
from parallel_wavegan.models import ParallelWaveGANDiscriminator
from parallel_wavegan.models import ResidualParallelWaveGANDiscriminator
from parallel_wavegan.optimizers import RAdam

from test_parallel_wavegan import make_discriminator_args
from test_parallel_wavegan import make_mutli_reso_stft_loss_args
from test_parallel_wavegan import make_residual_discriminator_args

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def make_melgan_generator_args(**kwargs):
    defaults = dict(
        in_channels=80,
        out_channels=1,
        kernel_size=7,
        channels=512,
        bias=True,
        upsample_scales=[8, 8, 2, 2],
        stack_kernel_size=3,
        stacks=3,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_final_nonlinear_activation=True,
        use_weight_norm=True,
        use_causal_conv=False,
    )
    defaults.update(kwargs)
    return defaults


def make_melgan_discriminator_args(**kwargs):
    defaults = dict(
        in_channels=1,
        out_channels=1,
        scales=3,
        downsample_pooling="AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "count_include_pad": False,
        },
        kernel_sizes=[5, 3],
        channels=16,
        max_downsample_channels=1024,
        bias=True,
        downsample_scales=[4, 4, 4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_weight_norm=True,
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss",
    [
        ({}, {}, {}),
        ({"kernel_size": 3}, {}, {}),
        ({"channels": 1024}, {}, {}),
        ({"stack_kernel_size": 5}, {}, {}),
        ({"stack_kernel_size": 5, "stacks": 2}, {}, {}),
        ({"upsample_scales": [4, 4, 4, 4]}, {}, {}),
        ({"upsample_scales": [8, 8, 2, 2, 2]}, {}, {}),
        ({"channels": 1024, "upsample_scales": [8, 8, 2, 2, 2, 2]}, {}, {}),
        ({"pad": "ConstantPad1d", "pad_params": {"value": 0.0}}, {}, {}),
        ({"nonlinear_activation": "ReLU", "nonlinear_activation_params": {}}, {}, {}),
        ({"bias": False}, {}, {}),
        ({"use_final_nonlinear_activation": False}, {}, {}),
        ({"use_weight_norm": False}, {}, {}),
        ({"use_causal_conv": True}, {}, {}),
    ],
)
def test_melgan_trainable(dict_g, dict_d, dict_loss):
    # setup
    batch_size = 4
    batch_length = 4096
    args_g = make_melgan_generator_args(**dict_g)
    args_d = make_discriminator_args(**dict_d)
    args_loss = make_mutli_reso_stft_loss_args(**dict_loss)
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(
        batch_size,
        args_g["in_channels"],
        batch_length // np.prod(args_g["upsample_scales"]),
    )
    model_g = MelGANGenerator(**args_g)
    model_d = ParallelWaveGANDiscriminator(**args_d)
    aux_criterion = MultiResolutionSTFTLoss(**args_loss)
    gen_adv_criterion = GeneratorAdversarialLoss()
    dis_adv_criterion = DiscriminatorAdversarialLoss()
    optimizer_g = RAdam(model_g.parameters())
    optimizer_d = RAdam(model_d.parameters())

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


@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss",
    [
        ({}, {}, {}),
        ({"kernel_size": 3}, {}, {}),
        ({"channels": 1024}, {}, {}),
        ({"stack_kernel_size": 5}, {}, {}),
        ({"stack_kernel_size": 5, "stacks": 2}, {}, {}),
        ({"upsample_scales": [4, 4, 4, 4]}, {}, {}),
        ({"upsample_scales": [8, 8, 2, 2, 2]}, {}, {}),
        ({"channels": 1024, "upsample_scales": [8, 8, 2, 2, 2, 2]}, {}, {}),
        ({"pad": "ConstantPad1d", "pad_params": {"value": 0.0}}, {}, {}),
        ({"nonlinear_activation": "ReLU", "nonlinear_activation_params": {}}, {}, {}),
        ({"bias": False}, {}, {}),
        ({"use_final_nonlinear_activation": False}, {}, {}),
        ({"use_weight_norm": False}, {}, {}),
    ],
)
def test_melgan_trainable_with_residual_discriminator(dict_g, dict_d, dict_loss):
    # setup
    batch_size = 4
    batch_length = 4096
    args_g = make_melgan_generator_args(**dict_g)
    args_d = make_residual_discriminator_args(**dict_d)
    args_loss = make_mutli_reso_stft_loss_args(**dict_loss)
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(
        batch_size,
        args_g["in_channels"],
        batch_length // np.prod(args_g["upsample_scales"]),
    )
    model_g = MelGANGenerator(**args_g)
    model_d = ResidualParallelWaveGANDiscriminator(**args_d)
    aux_criterion = MultiResolutionSTFTLoss(**args_loss)
    gen_adv_criterion = GeneratorAdversarialLoss()
    dis_adv_criterion = DiscriminatorAdversarialLoss()
    optimizer_g = RAdam(model_g.parameters())
    optimizer_d = RAdam(model_d.parameters())

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


@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss",
    [
        ({}, {}, {}),
        ({}, {"scales": 4}, {}),
        ({}, {"kernel_sizes": [7, 5]}, {}),
        ({}, {"max_downsample_channels": 128}, {}),
        ({}, {"downsample_scales": [4, 4]}, {}),
        ({}, {"pad": "ConstantPad1d", "pad_params": {"value": 0.0}}, {}),
        ({}, {"nonlinear_activation": "ReLU", "nonlinear_activation_params": {}}, {}),
    ],
)
def test_melgan_trainable_with_melgan_discriminator(dict_g, dict_d, dict_loss):
    # setup
    batch_size = 4
    batch_length = 4096
    args_g = make_melgan_generator_args(**dict_g)
    args_d = make_melgan_discriminator_args(**dict_d)
    args_loss = make_mutli_reso_stft_loss_args(**dict_loss)
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(
        batch_size,
        args_g["in_channels"],
        batch_length // np.prod(args_g["upsample_scales"]),
    )
    model_g = MelGANGenerator(**args_g)
    model_d = MelGANMultiScaleDiscriminator(**args_d)
    aux_criterion = MultiResolutionSTFTLoss(**args_loss)
    feat_match_criterion = FeatureMatchLoss()
    gen_adv_criterion = GeneratorAdversarialLoss()
    dis_adv_criterion = DiscriminatorAdversarialLoss()
    optimizer_g = RAdam(model_g.parameters())
    optimizer_d = RAdam(model_d.parameters())

    # check generator trainable
    y_hat = model_g(c)
    p_hat = model_d(y_hat)
    sc_loss, mag_loss = aux_criterion(y_hat, y)
    aux_loss = sc_loss + mag_loss
    adv_loss = gen_adv_criterion(p_hat)
    with torch.no_grad():
        p = model_d(y)
    fm_loss = feat_match_criterion(p_hat, p)
    loss_g = adv_loss + aux_loss + fm_loss
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


@pytest.mark.parametrize(
    "dict_g",
    [
        ({"use_causal_conv": True}),
        ({"use_causal_conv": True, "upsample_scales": [4, 4, 2, 2]}),
        ({"use_causal_conv": True, "upsample_scales": [4, 5, 4, 3]}),
    ],
)
def test_causal_melgan(dict_g):
    batch_size = 4
    batch_length = 4096
    args_g = make_melgan_generator_args(**dict_g)
    upsampling_factor = np.prod(args_g["upsample_scales"])
    c = torch.randn(
        batch_size, args_g["in_channels"], batch_length // upsampling_factor
    )
    model_g = MelGANGenerator(**args_g)
    c_ = c.clone()
    c_[..., c.size(-1) // 2 :] = torch.randn(c[..., c.size(-1) // 2 :].shape)
    try:
        # check not equal
        np.testing.assert_array_equal(c.numpy(), c_.numpy())
    except AssertionError:
        pass
    else:
        raise AssertionError("Must be different.")

    # check causality
    y = model_g(c)
    y_ = model_g(c_)
    assert y.size(2) == c.size(2) * upsampling_factor
    np.testing.assert_array_equal(
        y[..., : c.size(-1) // 2 * upsampling_factor].detach().cpu().numpy(),
        y_[..., : c_.size(-1) // 2 * upsampling_factor].detach().cpu().numpy(),
    )
