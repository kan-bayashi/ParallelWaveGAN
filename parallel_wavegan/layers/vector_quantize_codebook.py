# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Vector quantize codebook modules.

This code is modified from https://github.com/ritheshkumar95/pytorch-vqvae.

"""

import torch

from parallel_wavegan.functions import vector_quantize, vector_quantize_straight_through


class VQCodebook(torch.nn.Module):
    """Vector quantize codebook module."""

    def __init__(self, num_embeds, embed_dim):
        """Initialize VQCodebook module.

        Args:
            num_embeds (int): Number of embeddings.
            embed_dim (int): Dimension of each embedding.

        """
        super(VQCodebook, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeds, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeds, 1.0 / num_embeds)

    def forward(self, z_e):
        """Calculate forward propagation.

        Args:
            z_e (Tensor): Input tensor (B, embed_dim, T).

        Returns:
            LongTensor: Codebook indices (B, T).

        """
        z_e_ = z_e.transpose(2, 1).contiguous()
        indices = vector_quantize(z_e_, self.embedding.weight)

        return indices

    def straight_through(self, z_e):
        """Calculate forward propagation with straight through technique.

        Args:
            z_e (Tensor): Input tensor (B, embed_dim, T).

        Returns:
            Tensor: Codebook embeddings for the decoder inputs (B, embed_dim, T).
            Tensor: Codebook embeddings for the quantization loss (B, embed_dim, T).

        """
        # get embeddings for the decoder inputs
        z_e_ = z_e.transpose(2, 1).contiguous()
        z_q_, indices = vector_quantize_straight_through(
            z_e_, self.embedding.weight.detach()
        )
        z_q = z_q_.transpose(2, 1).contiguous()

        # get embedding for the quantization loss
        z_q_bar_flatten = torch.index_select(
            self.embedding.weight, dim=0, index=indices
        )
        z_q_bar_ = z_q_bar_flatten.view_as(z_e_)
        z_q_bar = z_q_bar_.transpose(1, 2).contiguous()

        return z_q, z_q_bar
