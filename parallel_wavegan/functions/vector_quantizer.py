# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Vector quantization modules.

These codes are modified from https://github.com/ritheshkumar95/pytorch-vqvae.

"""

import torch
from torch.autograd import Function


class VectorQuantization(Function):
    """Vector quantization modele."""

    @staticmethod
    @torch.no_grad()
    def forward(ctx, inputs, codebook):
        """Calculate forward propagation.

        Args:
            inputs (Tensor): Input tensor (B, `*`, embed_dim).
            codebook (Tensor): Embedding weights (num_embeds, embed_dim).

        Returns:
            LongTensor: Codebook indices (B, `*`).

        """
        embedding_size = codebook.size(1)
        inputs_size = inputs.size()
        inputs_flatten = inputs.view(-1, embedding_size)

        codebook_sqr = torch.sum(codebook**2, dim=1)
        inputs_sqr = torch.sum(inputs_flatten**2, dim=1, keepdim=True)

        # Compute the distances to the codebook
        distances = torch.addmm(
            codebook_sqr + inputs_sqr,
            inputs_flatten,
            codebook.t(),
            alpha=-2.0,
            beta=1.0,
        )

        _, indices_flatten = torch.min(distances, dim=1)
        indices = indices_flatten.view(*inputs_size[:-1])
        ctx.mark_non_differentiable(indices)

        return indices

    @staticmethod
    def backward(ctx, grad_output):
        """Calculate backward propagation."""
        raise RuntimeError(
            "Trying to call `.grad()` on graph containing "
            "`VectorQuantization`. The function `VectorQuantization` "
            "is not differentiable. Use `VectorQuantizationStraightThrough` "
            "if you want a straight-through estimator of the gradient."
        )


class VectorQuantizationStraightThrough(Function):
    """Differentiable vector quantize module with straight through technique."""

    @staticmethod
    def forward(ctx, inputs, codebook):
        """Calculate forward propagation.

        Args:
            inputs (Tensor): Input tensor (B, `*`, embed_dim).
            codebook (Tensor): Embedding weights (num_embeds, embed_dim).

        Returns:
            Tensor: Codebook embeddings (B, `*`, embed_dim).
            LongTensor: Codebook indices (B, `*`).

        """
        indices = vector_quantize(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        """Calculate backward propagation."""
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = grad_output.contiguous().view(-1, embedding_size)
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


# register functions
vector_quantize = VectorQuantization.apply
vector_quantize_straight_through = VectorQuantizationStraightThrough.apply
__all__ = [vector_quantize, vector_quantize_straight_through]
