#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

import logging

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from parallel_wavegan.losses import MultiResolutionSTFTLoss
from parallel_wavegan.models import MelGANGenerator
from parallel_wavegan.models import ParallelWaveGANDiscriminator
from parallel_wavegan.models import ResidualParallelWaveGANDiscriminator
from parallel_wavegan.optimizers import RAdam

from test_parallel_wavegan import make_discriminator_args
from test_parallel_wavegan import make_mutli_reso_stft_loss_args
from test_parallel_wavegan import make_residual_discriminator_args

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")


def make_generator_args(**kwargs):
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
        use_final_nolinear_activation=True,
        use_weight_norm=True,
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss", [
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
        ({"use_final_nolinear_activation": False}, {}, {}),
        ({"use_weight_norm": False}, {}, {}),
    ])
def test_melgan_trainable(dict_g, dict_d, dict_loss):
    # setup
    batch_size = 4
    batch_length = 4096
    args_g = make_generator_args(**dict_g)
    args_d = make_discriminator_args(**dict_d)
    args_loss = make_mutli_reso_stft_loss_args(**dict_loss)
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(batch_size, args_g["in_channels"],
                    batch_length // np.prod(
                        args_g["upsample_scales"]))
    model_g = MelGANGenerator(**args_g)
    model_d = ParallelWaveGANDiscriminator(**args_d)
    aux_criterion = MultiResolutionSTFTLoss(**args_loss)
    optimizer_g = RAdam(model_g.parameters())
    optimizer_d = RAdam(model_d.parameters())

    # check generator trainable
    y_hat = model_g(c)
    p_hat = model_d(y_hat)
    y, y_hat, p_hat = y.squeeze(1), y_hat.squeeze(1), p_hat.squeeze(1)
    adv_loss = F.mse_loss(p_hat, p_hat.new_ones(p_hat.size()))
    sc_loss, mag_loss = aux_criterion(y_hat, y)
    aux_loss = sc_loss + mag_loss
    loss_g = adv_loss + aux_loss
    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    # check discriminator trainable
    y, y_hat = y.unsqueeze(1), y_hat.unsqueeze(1).detach()
    p = model_d(y)
    p_hat = model_d(y_hat)
    p, p_hat = p.squeeze(1), p_hat.squeeze(1)
    loss_d = F.mse_loss(p, p.new_ones(p.size())) + F.mse_loss(p_hat, p_hat.new_zeros(p_hat.size()))
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()


@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss", [
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
        ({"use_final_nolinear_activation": False}, {}, {}),
        ({"use_weight_norm": False}, {}, {}),
    ])
def test_melgan_trainable_with_residual_discriminator(dict_g, dict_d, dict_loss):
    # setup
    batch_size = 4
    batch_length = 4096
    args_g = make_generator_args(**dict_g)
    args_d = make_residual_discriminator_args(**dict_d)
    args_loss = make_mutli_reso_stft_loss_args(**dict_loss)
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(batch_size, args_g["in_channels"],
                    batch_length // np.prod(
                        args_g["upsample_scales"]))
    model_g = MelGANGenerator(**args_g)
    model_d = ResidualParallelWaveGANDiscriminator(**args_d)
    aux_criterion = MultiResolutionSTFTLoss(**args_loss)
    optimizer_g = RAdam(model_g.parameters())
    optimizer_d = RAdam(model_d.parameters())

    # check generator trainable
    y_hat = model_g(c)
    p_hat = model_d(y_hat)
    y, y_hat, p_hat = y.squeeze(1), y_hat.squeeze(1), p_hat.squeeze(1)
    adv_loss = F.mse_loss(p_hat, p_hat.new_ones(p_hat.size()))
    sc_loss, mag_loss = aux_criterion(y_hat, y)
    aux_loss = sc_loss + mag_loss
    loss_g = adv_loss + aux_loss
    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    # check discriminator trainable
    y, y_hat = y.unsqueeze(1), y_hat.unsqueeze(1).detach()
    p = model_d(y)
    p_hat = model_d(y_hat)
    p, p_hat = p.squeeze(1), p_hat.squeeze(1)
    loss_d = F.mse_loss(p, p.new_ones(p.size())) + F.mse_loss(p_hat, p_hat.new_zeros(p_hat.size()))
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()
