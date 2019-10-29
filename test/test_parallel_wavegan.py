#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

from parallel_wavegan.losses import MultiResolutionSTFTLoss
from parallel_wavegan.models import ParallelWaveGANDiscriminator
from parallel_wavegan.models import ParallelWaveGANGenerator
from parallel_wavegan.optimizers import RAdam


def test_parallel_wavegan_trainable():
    # setup
    batch_size = 4
    batch_length = 4096
    z = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(batch_size, 80, batch_length // 256)
    y = torch.randn(batch_size, 1, batch_length)
    generator = ParallelWaveGANGenerator()
    discriminator = ParallelWaveGANDiscriminator()
    aux_criterion = MultiResolutionSTFTLoss()
    optimizer_g = RAdam(generator.parameters())
    optimizer_d = RAdam(discriminator.parameters())

    # check generator trainable
    y_hat = generator(z, c)
    p_hat = discriminator(y_hat)
    y, y_hat, p_hat = y.squeeze(1), y_hat.squeeze(1), p_hat.squeeze(1)
    adv_loss = F.mse_loss(p_hat, p_hat.new_ones(p_hat.size()))
    aux_loss = aux_criterion(y_hat, y)
    loss_g = adv_loss + aux_loss
    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    # check discriminator trainable
    y, y_hat = y.unsqueeze(1), y_hat.unsqueeze(1).detach()
    p = discriminator(y)
    p_hat = discriminator(y_hat)
    p, p_hat = p.squeeze(1), p_hat.squeeze(1)
    loss_d = F.mse_loss(p, p.new_ones(p.size())) + F.mse_loss(p_hat, p_hat.new_zeros(p_hat.size()))
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()
