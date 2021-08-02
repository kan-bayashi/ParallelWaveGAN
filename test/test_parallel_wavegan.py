#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

import logging

import numpy as np
import pytest
import torch

from parallel_wavegan.losses import DiscriminatorAdversarialLoss
from parallel_wavegan.losses import GeneratorAdversarialLoss
from parallel_wavegan.losses import MultiResolutionSTFTLoss
from parallel_wavegan.models import ParallelWaveGANDiscriminator
from parallel_wavegan.models import ParallelWaveGANGenerator
from parallel_wavegan.models import ResidualParallelWaveGANDiscriminator
from parallel_wavegan.optimizers import RAdam

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def make_generator_args(**kwargs):
    defaults = dict(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        layers=6,
        stacks=3,
        residual_channels=8,
        gate_channels=16,
        skip_channels=8,
        aux_channels=10,
        aux_context_window=0,
        dropout=1 - 0.95,
        use_weight_norm=True,
        use_causal_conv=False,
        upsample_conditional_features=True,
        upsample_net="ConvInUpsampleNetwork",
        upsample_params={"upsample_scales": [4, 4]},
    )
    defaults.update(kwargs)
    return defaults


def make_discriminator_args(**kwargs):
    defaults = dict(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        layers=5,
        conv_channels=16,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        bias=True,
        use_weight_norm=True,
    )
    defaults.update(kwargs)
    return defaults


def make_residual_discriminator_args(**kwargs):
    defaults = dict(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        layers=10,
        stacks=1,
        residual_channels=8,
        gate_channels=16,
        skip_channels=8,
        dropout=0.0,
        use_weight_norm=True,
        use_causal_conv=False,
        nonlinear_activation_params={"negative_slope": 0.2},
    )
    defaults.update(kwargs)
    return defaults


def make_mutli_reso_stft_loss_args(**kwargs):
    defaults = dict(
        fft_sizes=[64, 128, 256],
        hop_sizes=[32, 64, 128],
        win_lengths=[48, 96, 192],
        window="hann_window",
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss",
    [
        ({}, {}, {}),
        ({"layers": 1, "stacks": 1}, {}, {}),
        ({}, {"layers": 1}, {}),
        ({"kernel_size": 5}, {}, {}),
        ({}, {"kernel_size": 5}, {}),
        ({"gate_channels": 8}, {}, {}),
        ({"stacks": 1}, {}, {}),
        ({"use_weight_norm": False}, {"use_weight_norm": False}, {}),
        ({"aux_context_window": 2}, {}, {}),
        ({"upsample_net": "UpsampleNetwork"}, {}, {}),
        (
            {"upsample_params": {"upsample_scales": [4], "freq_axis_kernel_size": 3}},
            {},
            {},
        ),
        (
            {
                "upsample_params": {
                    "upsample_scales": [4],
                    "nonlinear_activation": "ReLU",
                }
            },
            {},
            {},
        ),
        (
            {
                "upsample_conditional_features": False,
                "upsample_params": {"upsample_scales": [1]},
            },
            {},
            {},
        ),
        ({}, {"nonlinear_activation": "ReLU", "nonlinear_activation_params": {}}, {}),
        ({"use_causal_conv": True}, {}, {}),
        ({"use_causal_conv": True, "upsample_net": "UpsampleNetwork"}, {}, {}),
        ({"use_causal_conv": True, "aux_context_window": 1}, {}, {}),
        ({"use_causal_conv": True, "aux_context_window": 2}, {}, {}),
        ({"use_causal_conv": True, "aux_context_window": 3}, {}, {}),
        (
            {
                "aux_channels": 16,
                "upsample_net": "MelGANGenerator",
                "upsample_params": {
                    "upsample_scales": [4, 4],
                    "in_channels": 16,
                    "out_channels": 16,
                },
            },
            {},
            {},
        ),
    ],
)
def test_parallel_wavegan_trainable(dict_g, dict_d, dict_loss):
    # setup
    batch_size = 4
    batch_length = 4096
    args_g = make_generator_args(**dict_g)
    args_d = make_discriminator_args(**dict_d)
    args_loss = make_mutli_reso_stft_loss_args(**dict_loss)
    z = torch.randn(batch_size, 1, batch_length)
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(
        batch_size,
        args_g["aux_channels"],
        batch_length // np.prod(args_g["upsample_params"]["upsample_scales"])
        + 2 * args_g["aux_context_window"],
    )
    model_g = ParallelWaveGANGenerator(**args_g)
    model_d = ParallelWaveGANDiscriminator(**args_d)
    aux_criterion = MultiResolutionSTFTLoss(**args_loss)
    gen_adv_criterion = GeneratorAdversarialLoss()
    dis_adv_criterion = DiscriminatorAdversarialLoss()
    optimizer_g = RAdam(model_g.parameters())
    optimizer_d = RAdam(model_d.parameters())

    # check generator trainable
    y_hat = model_g(z, c)
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
        ({"layers": 1, "stacks": 1}, {}, {}),
        ({}, {"layers": 1}, {}),
        ({"kernel_size": 5}, {}, {}),
        ({}, {"kernel_size": 5}, {}),
        ({"gate_channels": 8}, {}, {}),
        ({"stacks": 1}, {}, {}),
        ({"use_weight_norm": False}, {"use_weight_norm": False}, {}),
        ({"aux_context_window": 2}, {}, {}),
        ({"upsample_net": "UpsampleNetwork"}, {}, {}),
        (
            {"upsample_params": {"upsample_scales": [4], "freq_axis_kernel_size": 3}},
            {},
            {},
        ),
        (
            {
                "upsample_params": {
                    "upsample_scales": [4],
                    "nonlinear_activation": "ReLU",
                }
            },
            {},
            {},
        ),
        (
            {
                "upsample_conditional_features": False,
                "upsample_params": {"upsample_scales": [1]},
            },
            {},
            {},
        ),
        ({}, {"nonlinear_activation": "ReLU", "nonlinear_activation_params": {}}, {}),
        ({"use_causal_conv": True}, {}, {}),
        ({"use_causal_conv": True, "upsample_net": "UpsampleNetwork"}, {}, {}),
        ({"use_causal_conv": True, "aux_context_window": 1}, {}, {}),
        ({"use_causal_conv": True, "aux_context_window": 2}, {}, {}),
        ({"use_causal_conv": True, "aux_context_window": 3}, {}, {}),
        (
            {
                "aux_channels": 16,
                "upsample_net": "MelGANGenerator",
                "upsample_params": {
                    "upsample_scales": [4, 4],
                    "in_channels": 16,
                    "out_channels": 16,
                },
            },
            {},
            {},
        ),
    ],
)
def test_parallel_wavegan_with_residual_discriminator_trainable(
    dict_g, dict_d, dict_loss
):
    # setup
    batch_size = 4
    batch_length = 4096
    args_g = make_generator_args(**dict_g)
    args_d = make_residual_discriminator_args(**dict_d)
    args_loss = make_mutli_reso_stft_loss_args(**dict_loss)
    z = torch.randn(batch_size, 1, batch_length)
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(
        batch_size,
        args_g["aux_channels"],
        batch_length // np.prod(args_g["upsample_params"]["upsample_scales"])
        + 2 * args_g["aux_context_window"],
    )
    model_g = ParallelWaveGANGenerator(**args_g)
    model_d = ResidualParallelWaveGANDiscriminator(**args_d)
    aux_criterion = MultiResolutionSTFTLoss(**args_loss)
    gen_adv_criterion = GeneratorAdversarialLoss()
    dis_adv_criterion = DiscriminatorAdversarialLoss()
    optimizer_g = RAdam(model_g.parameters())
    optimizer_d = RAdam(model_d.parameters())

    # check generator trainable
    y_hat = model_g(z, c)
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
    "upsample_net, aux_context_window",
    [
        ("ConvInUpsampleNetwork", 0),
        ("ConvInUpsampleNetwork", 1),
        ("ConvInUpsampleNetwork", 2),
        ("ConvInUpsampleNetwork", 3),
        ("UpsampleNetwork", 0),
    ],
)
def test_causal_parallel_wavegan(upsample_net, aux_context_window):
    batch_size = 1
    batch_length = 4096
    args_g = make_generator_args(
        use_causal_conv=True,
        upsample_net=upsample_net,
        aux_context_window=aux_context_window,
        dropout=0.0,
    )
    model_g = ParallelWaveGANGenerator(**args_g)
    z = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(
        batch_size,
        args_g["aux_channels"],
        batch_length // np.prod(args_g["upsample_params"]["upsample_scales"]),
    )

    z_ = z.clone()
    c_ = c.clone()
    z_[..., z.size(-1) // 2 :] = torch.randn(z[..., z.size(-1) // 2 :].shape)
    c_[..., c.size(-1) // 2 :] = torch.randn(c[..., c.size(-1) // 2 :].shape)
    c = torch.nn.ConstantPad1d(args_g["aux_context_window"], 0.0)(c)
    c_ = torch.nn.ConstantPad1d(args_g["aux_context_window"], 0.0)(c_)
    try:
        # check not equal
        np.testing.assert_array_equal(c.numpy(), c_.numpy())
    except AssertionError:
        pass
    else:
        raise AssertionError("Must be different.")
    try:
        # check not equal
        np.testing.assert_array_equal(z.numpy(), z_.numpy())
    except AssertionError:
        pass
    else:
        raise AssertionError("Must be different.")

    # check causality
    y = model_g(z, c)
    y_ = model_g(z_, c_)
    np.testing.assert_array_equal(
        y[..., : y.size(-1) // 2].detach().cpu().numpy(),
        y_[..., : y_.size(-1) // 2].detach().cpu().numpy(),
    )
