# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""VQVAE Modules."""

import logging

import torch

import parallel_wavegan.models
from parallel_wavegan.layers import VQCodebook


class VQVAE(torch.nn.Module):
    """VQVAE module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_embeds=512,
        embed_dim=256,
        num_local_embeds=None,
        local_embed_dim=None,
        num_global_embeds=None,
        global_embed_dim=None,
        encoder_type="MelGANDiscriminator",
        decoder_type="MelGANGenerator",
        encoder_conf={
            "out_channels": 256,
            "downsample_scales": [4, 4, 2, 2],
            "max_downsample_channels": 1024,
        },
        decoder_conf={
            "in_channels": 256,
            "upsample_scales": [4, 4, 2, 2],
            "channels": 512,
            "stacks": 3,
        },
        use_weight_norm=True,
    ):
        """Initialize VQVAE module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_embeds (int): Number of embeddings.
            embed_dim (int): Dimension of each embedding.
            num_local_embeds (int): Number of local embeddings.
            local_embed_dim (int): Dimension of each local embedding.
            num_global_embeds (int): Number of global embeddings.
            global_embed_dim (int): Dimension of each global embedding.
            encoder_type (str): Encoder module name.
            decoder_type (str): Decoder module name.
            encoder_conf (dict): Hyperparameters for the encoder.
            decoder_conf (dict): Hyperparameters for the decoder.
            use_weight_norm (bool): Whether to use weight norm.

        """
        super(VQVAE, self).__init__()
        encoder_class = getattr(parallel_wavegan.models, encoder_type)
        decoder_class = getattr(parallel_wavegan.models, decoder_type)
        encoder_conf.update({"in_channels": in_channels})
        decoder_conf.update({"out_channels": out_channels})
        if not issubclass(decoder_class, parallel_wavegan.models.MelGANGenerator):
            raise NotImplementedError(f"{decoder_class} is not supported yet.")
        if num_local_embeds is not None:
            if local_embed_dim is not None:
                self.local_embed = torch.nn.Conv1d(num_local_embeds, local_embed_dim, 1)
            else:
                self.local_embed = None
        if num_global_embeds is not None:
            self.global_embed = torch.nn.Embedding(num_global_embeds, global_embed_dim)
        self.encoder = encoder_class(**encoder_conf)
        self.codebook = VQCodebook(num_embeds=num_embeds, embed_dim=embed_dim)
        self.decoder = decoder_class(**decoder_conf)

        # apply weight norm
        if use_weight_norm:
            self.remove_weight_norm()  # for duplicated weight norm
            self.apply_weight_norm()

    def forward(self, x, l=None, g=None):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            l (Tensor): Local conditioning tensor (B, num_local_embeds, T).
            g (LongTensor): Global conditioning idx (B, ).

        Return:
            Tensor: Reconstruced input tensor (B, in_channels, T).
            Tensor: Encoder hidden states (B, embed_dim, T // prod(downsample_scales)).
            Tensor: Quantized encoder hidden states (B, embed_dim, T // prod(downsample_scales)).

        """
        z_e = self.encoder(x)
        z_e = z_e[-1] if isinstance(z_e, list) else z_e  # For MelGAN Discriminator
        z_q_st, z_q = self.codebook.straight_through(z_e)
        if l is not None:
            if self.local_embed is not None:
                l = self.local_embed(l)
            z_q_st = torch.cat([z_q_st, l], dim=1)
        if g is not None:
            g = self.global_embed(g).unsqueeze(2).expand(-1, -1, z_q_st.size(2))
            z_q_st = torch.cat([z_q_st, g], dim=1)
        x_bar = self.decoder(z_q_st)

        return x_bar, z_e, z_q

    def encode(self, x):
        """Encode the inputs into the latent codes.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).

        Returns:
            LongTensor: Quantized tensor (B, T).

        """
        z_e = self.encoder(x)[-1]
        z_e = z_e[-1] if isinstance(z_e, list) else z_e  # For MelGAN Discriminator
        return self.codebook(z_e)

    def decode(self, indices, l=None, g=None):
        """Decode the latent codes to the inputs.

        Args:
            indices (LongTensor): Quantized tensor (B, T).
            l (Tensor): Local conditioning tensor (B, num_local_embeds, T).
            g (LongTensor): Global conditioning idx (B, ).

        Return:
            Tensor: Reconstruced tensor (B, 1, T).

        """
        z_q = self.codebook.embedding(indices).transpose(2, 1)
        if l is not None:
            if self.local_embed is not None:
                l = self.local_embed(l)
            z_q = torch.cat([z_q, l], dim=1)
        if g is not None:
            g = self.global_embed(g).unsqueeze(2).expand(-1, -1, z_q.size(2))
            z_q = torch.cat([z_q, g], dim=1)
        return self.decoder(z_q)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)
