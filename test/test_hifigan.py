#!/usr/bin/env python3

# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Test code for HiFi-GAN modules."""

import logging

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from parallel_wavegan.losses import MultiResolutionSTFTLoss
from parallel_wavegan.models import HiFiGANGenerator
from parallel_wavegan.models import HiFiGANMultiScaleMultiPeriodDiscriminator
from test_parallel_wavegan import make_mutli_reso_stft_loss_args


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def make_hifigan_generator_args(**kwargs):
    defaults = dict(
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernal_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
    )
    defaults.update(kwargs)
    return defaults


def make_hifigan_multi_scale_multi_period_discriminator_args(**kwargs):
    defaults = dict(
        scales=3,
        scale_downsample_pooling="AvgPool1d",
        scale_downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        scale_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 16,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [4, 4, 4, 4],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm=False,
        periods=[2, 3, 5, 7, 11],
        period_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [4, 4, 4, 4],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss",
    [
        ({}, {}, {}),
        ({}, {"scales": 1}, {}),
        ({}, {"periods": [2]}, {}),
        ({}, {"follow_official_norm": True}, {}),
        ({"use_additional_convs": False}, {}, {}),
    ],
)
def test_hifigan_trainable(dict_g, dict_d, dict_loss):
    # setup
    batch_size = 4
    batch_length = 2 ** 13
    args_g = make_hifigan_generator_args(**dict_g)
    args_d = make_hifigan_multi_scale_multi_period_discriminator_args(**dict_d)
    args_loss = make_mutli_reso_stft_loss_args(**dict_loss)
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(
        batch_size,
        args_g["in_channels"],
        batch_length // np.prod(args_g["upsample_scales"]),
    )
    model_g = HiFiGANGenerator(**args_g)
    model_d = HiFiGANMultiScaleMultiPeriodDiscriminator(**args_d)
    aux_criterion = MultiResolutionSTFTLoss(**args_loss)
    optimizer_g = torch.optim.AdamW(model_g.parameters())
    optimizer_d = torch.optim.AdamW(model_d.parameters())

    # check generator trainable
    y_hat = model_g(c)
    p_hat_ms, p_hat_mp = model_d(y_hat)
    y, y_hat = y.squeeze(1), y_hat.squeeze(1)
    sc_loss, mag_loss = aux_criterion(y_hat, y)
    aux_loss = sc_loss + mag_loss
    adv_loss = 0.0
    for i_ms in range(len(p_hat_ms)):
        adv_loss += F.mse_loss(
            p_hat_ms[i_ms][-1], p_hat_ms[i_ms][-1].new_ones(p_hat_ms[i_ms][-1].size())
        )
    for i_mp in range(len(p_hat_mp)):
        adv_loss += F.mse_loss(
            p_hat_mp[i_mp][-1], p_hat_mp[i_mp][-1].new_ones(p_hat_mp[i_mp][-1].size())
        )
    with torch.no_grad():
        p_ms, p_mp = model_d(y.unsqueeze(1))
    fm_loss = 0.0
    for i in range(len(p_hat_ms)):
        for j in range(len(p_hat_ms[i]) - 1):
            fm_loss += F.l1_loss(p_hat_ms[i][j], p_ms[i][j].detach())
    for i in range(len(p_hat_mp)):
        for j in range(len(p_hat_mp[i]) - 1):
            fm_loss += F.l1_loss(p_hat_mp[i][j], p_mp[i][j].detach())
    loss_g = adv_loss + aux_loss + fm_loss
    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    # check discriminator trainable
    y, y_hat = y.unsqueeze(1), y_hat.unsqueeze(1).detach()
    p_ms, p_mp = model_d(y)
    p_hat_ms, p_hat_mp = model_d(y_hat)
    real_loss = 0.0
    fake_loss = 0.0
    for i in range(len(p_ms)):
        real_loss += F.mse_loss(p_ms[i][-1], p_ms[i][-1].new_ones(p_ms[i][-1].size()))
        fake_loss += F.mse_loss(
            p_hat_ms[i][-1], p_hat_ms[i][-1].new_zeros(p_hat_ms[i][-1].size())
        )
    for i in range(len(p_mp)):
        real_loss += F.mse_loss(p_mp[i][-1], p_mp[i][-1].new_ones(p_mp[i][-1].size()))
        fake_loss += F.mse_loss(
            p_hat_mp[i][-1], p_hat_mp[i][-1].new_zeros(p_hat_mp[i][-1].size())
        )
    loss_d = real_loss + fake_loss
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()
