#!/usr/bin/env python3

# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Test code for HiFi-GAN modules."""

import logging
import os

import numpy as np
import pytest
import torch
import yaml
from test_parallel_wavegan import make_mutli_reso_stft_loss_args

import parallel_wavegan.models
from parallel_wavegan.losses import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
    MultiResolutionSTFTLoss,
)
from parallel_wavegan.models import (
    HiFiGANGenerator,
    HiFiGANMultiScaleMultiPeriodDiscriminator,
)

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
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_causal_conv=False,
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
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 128,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
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
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 128,
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
        ({}, {"scales": 1, "periods": [2]}, {}),
        ({}, {"follow_official_norm": True}, {}),
        ({"use_additional_convs": False}, {}, {}),
    ],
)
def test_hifigan_trainable(dict_g, dict_d, dict_loss):
    # setup
    batch_size = 4
    batch_length = 2**13
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
    feat_match_criterion = FeatureMatchLoss(
        average_by_layers=False,
        average_by_discriminators=False,
        include_final_outputs=True,
    )
    gen_adv_criterion = GeneratorAdversarialLoss(
        average_by_discriminators=False,
    )
    dis_adv_criterion = DiscriminatorAdversarialLoss(
        average_by_discriminators=False,
    )
    optimizer_g = torch.optim.AdamW(model_g.parameters())
    optimizer_d = torch.optim.AdamW(model_d.parameters())

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

    print(model_d)
    print(model_g)


@pytest.mark.parametrize(
    "dict_g",
    [
        (
            {
                "use_causal_conv": True,
                "upsample_scales": [5, 5, 4, 3],
                "upsample_kernel_sizes": [10, 10, 8, 6],
            }
        ),
        (
            {
                "use_causal_conv": True,
                "upsample_scales": [8, 8, 2, 2],
                "upsample_kernel_sizes": [16, 16, 4, 4],
            }
        ),
        (
            {
                "use_causal_conv": True,
                "upsample_scales": [4, 5, 4, 3],
                "upsample_kernel_sizes": [8, 10, 8, 6],
            }
        ),
        (
            {
                "use_causal_conv": True,
                "upsample_scales": [4, 4, 2, 2],
                "upsample_kernel_sizes": [8, 8, 4, 4],
            }
        ),
    ],
)
def test_causal_hifigan(dict_g):
    batch_size = 4
    batch_length = 8192
    args_g = make_hifigan_generator_args(**dict_g)
    upsampling_factor = np.prod(args_g["upsample_scales"])
    c = torch.randn(
        batch_size, args_g["in_channels"], batch_length // upsampling_factor
    )
    model_g = HiFiGANGenerator(**args_g)
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


def test_fix_norm_issue():
    from parallel_wavegan.utils import download_pretrained_model

    checkpoint = download_pretrained_model("ljspeech_hifigan.v1")
    config = os.path.join(os.path.dirname(checkpoint), "config.yml")
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # get model and load parameters
    discriminator_type = config.get("discriminator_type")
    model_class = getattr(
        parallel_wavegan.models,
        discriminator_type,
    )
    model = model_class(**config["discriminator_params"])

    state_dict_org = model.state_dict()
    model.load_state_dict(state_dict_org)

    state_dict = torch.load(checkpoint, map_location="cpu")["model"]["discriminator"]
    model.load_state_dict(state_dict, strict=False)
