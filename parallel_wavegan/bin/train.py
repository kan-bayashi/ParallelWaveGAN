#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Parallel WaveGAN."""

import argparse

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader

from parallel_wavegan.losses import MultiResolutionSTFTLoss
from parallel_wavegan.models import ParallelWaveGANDiscriminator
from parallel_wavegan.models import ParallelWaveGANGenerator
from parallel_wavegan.optimizers import RAdam
from parallel_wavegan.utils.dataset import PyTorchDataset


class CustomCollater(object):
    """Customized collater for Pytorch DataLoader."""

    def __init__(self,
                 batch_max_steps=20480,
                 hop_size=256,
                 aux_context_window=2,
                 device=torch.device("cpu")
                 ):
        """Initialize customized collater."""
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.device = device

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T").
            Tensor: Target signal batch (B, 1, T).
            LongTensor: Input length batch (B,)

        """
        # Time resolution adjustment
        new_batch = []
        for idx in range(len(batch)):
            x, c = batch[idx]
            self._assert_ready_for_upsampling(x, c, self.hop_size, 0)
            if len(x) > self.batch_max_steps:
                interval_start = self.aux_context_window
                interval_end = len(c) - self.batch_max_frames - self.aux_context_window
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * self.hop_size
                x = x[start_step: start_step + self.batch_max_steps]
                c = c[start_frame - self.aux_context_window:
                      start_frame + self.aux_context_window + self.batch_max_frames]
                self._assert_ready_for_upsampling(x, c, self.hop_size, self.aux_context_window)
            new_batch.append((x, c))
        batch = new_batch

        # Make padded target signale batch
        xlens = [len(b[0]) for b in batch]
        max_olen = max(xlens)
        y_batch = np.array([self._pad_2darray(b[0].reshape(-1, 1), max_olen) for b in batch], dtype=np.float32)
        y_batch = torch.FloatTensor(y_batch).transpose(2, 1).to(self.device)

        # Make padded conditional auxiliary feature batch
        clens = [len(b[1]) for b in batch]
        max_clen = max(clens)
        c_batch = np.array([self._pad_2darray(b[1], max_clen) for b in batch], dtype=np.float32)
        c_batch = torch.FloatTensor(c_batch).transpose(2, 1).to(self.device)

        # Make input noise signale batch
        z_batch = torch.randn(y_batch.size()).to(self.device)

        # Make the list of the length of input signals
        input_lengths = torch.LongTensor(xlens).to(self.device)

        return z_batch, c_batch, y_batch, input_lengths

    @staticmethod
    def _assert_ready_for_upsampling(x, c, hop_size, context_window):
        assert len(x) == (len(c) - 2 * context_window) * hop_size

    @staticmethod
    def _pad_2darray(x, max_len, b_pad=0, constant_values=0):
        return np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
                      mode="constant", constant_values=constant_values)


def main():
    """Run main process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dumpdir", default=None, type=str,
                        help="Directory including trainning data.")
    parser.add_argument("--dev-dumpdir", default=None, type=str,
                        help="Direcotry including development data.")
    parser.add_argument("--config", default="hparam.yml", type=str,
                        help="Yaml format configuration file.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # get dataset
    train_dataset = PyTorchDataset(args.train_dumpdir)
    dev_dataset = PyTorchDataset(args.dev_dumpdir)

    # get data loader
    collate_fn = CustomCollater()
    data_loader = {}
    data_loader["train"] = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn)
    data_loader["eval"] = DataLoader(dev_dataset, shuffle=True, collate_fn=collate_fn)

    # define models and optimizers
    model_g = ParallelWaveGANGenerator(**config["generator"])
    model_d = ParallelWaveGANDiscriminator(**config["discriminator"])
    stft_criterion = MultiResolutionSTFTLoss(**config["stft_loss"])
    mse_criterion = torch.nn.MSELoss()
    optimizer_g = RAdam(model_g.parameters(), lr=config["generator_lr"], eps=config["eps"])
    optimizer_d = RAdam(model_d.parameters(), lr=config["discriminator_lr"], eps=config["eps"])

    for z, c, y, input_lengths in data_loader["train"]:
        y_hat = model_g(z, c)
        p_hat = model_d(y_hat)
        y, y_hat, p_hat = y.squeeze(1), y_hat.squeeze(1), p_hat.squeeze(1)
        adv_loss = mse_criterion(p_hat, p_hat.new_ones(p_hat.size()))
        aux_loss = stft_criterion(y_hat, y)
        loss_g = adv_loss + config["lambda_adv"] * aux_loss
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        y, y_hat = y.unsqueeze(1), y_hat.unsqueeze(1).detach()
        p = model_d(y)
        p_hat = model_d(y_hat)
        p, p_hat = p.squeeze(1), p_hat.squeeze(1)
        loss_d = mse_criterion(p, p.new_ones(p.size())) + mse_criterion(p_hat, p_hat.new_zeros(p_hat.size()))
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()


if __name__ == "__main__":
    main()
