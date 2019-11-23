# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Mu-Law conversion modules."""

import torch


class MuLawEncode(torch.nn.Module):
    """Mu-law encoding module."""

    def __init__(self, mu=65535):
        """Initialize Mu-law encoding module.

        Args:
            mu (int): Quantized level. Must be an expornent of 2 - 1.

        """
        super(MuLawEncode, self).__init__()
        self.mu = mu

    def forward(self, x):
        """Caluclate forward propagation.

        Args:
            x (Tensor): Audio signal with the range from -1 to 1.

        Returns:
            Tensor: Quantized audio signal with the range from -1 to 1.

        """
        return mu_law_encode(x, self.mu)


class MuLawDecode(torch.nn.Module):
    """Mu-law decoding module."""

    def __init__(self, mu=65535):
        """Initialize Mu-law decoding module.

        Args:
            mu (int): Quantized level. Must be an expornent of 2 - 1.

        """
        super(MuLawDecode, self).__init__()
        self.mu = mu

    def forward(self, x):
        """Caluclate forward propagation.

        Args:
            x (torch): Quantized audio signal with the range from -1 to 1.

        Returns:
            Tensor: Audio signal with the range from -1 to 1.

        """
        return mu_law_decode(x, self.mu)


def mu_law_encode(x, mu=65535):
    """Perform mu-law encoding.

    Args:
        x (Tensor): Audio signal with the range from -1 to 1.
        mu (int): Quantized level.

    Returns:
        Tensor: Quantized audio signal with the range from -1 to 1.

    """
    assert (mu + 1) % 2 == 0, "mu must be an expornent of 2 - 1."
    mu = x.new_tensor(mu)
    return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)


def mu_law_decode(x, mu=65535):
    """Perform mu-law decoding.

    Args:
        x (torch): Quantized audio signal with the range from -1 to 1.
        mu (int): Quantized level.

    Returns:
        Tensor: Audio signal with the range from -1 to 1.

    """
    assert (mu + 1) % 2 == 0, "mu must be an expornent of 2 - 1."
    mu = x.new_tensor(mu)
    return torch.sign(x) / mu * ((1 + mu) ** torch.abs(x) - 1)
