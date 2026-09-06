"""Explicit scaled dot-product causal attention (chapter 3 concepts)."""

import math

import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(self, width: int, heads: int, context_length: int, dropout: float = 0.0):
        super().__init__()
        if width < 1 or heads < 1 or width % heads:
            raise ValueError("width must be positive and divisible by positive heads")
        if context_length < 1:
            raise ValueError("context_length must be positive")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "future_mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        batch, length, width = x.shape
        if not 1 <= length <= self.future_mask.size(0):
            raise ValueError("Sequence length must be within the configured context")
        # [B,T,3C] -> three tensors [B,H,T,D].
        q, k, v = (
            self.qkv(x)
            .reshape(batch, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.future_mask[:length, :length], float("-inf"))
        weights = self.dropout(scores.softmax(dim=-1))
        context = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.projection(context)
