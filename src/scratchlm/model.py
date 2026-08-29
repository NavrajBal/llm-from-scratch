"""Pre-normalized decoder-only transformer (chapter 4 concepts)."""

from dataclasses import dataclass

import torch
from torch import nn

from scratchlm.attention import CausalAttention


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 258
    context_length: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.width, self.heads, self.layers) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class LayerNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.shift = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) * torch.rsqrt(variance + self.eps) + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x**3)))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = LayerNorm(config.width)
        self.attention = CausalAttention(
            config.width, config.heads, config.context_length, config.dropout
        )
        self.norm2 = LayerNorm(config.width)
        self.feedforward = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            GELU(),
            nn.Linear(4 * config.width, config.width),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.dropout(self.feedforward(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.layers)))
        self.final_norm = LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def hidden_states(self, tokens):
        if tokens.ndim != 2 or not 1 <= tokens.size(1) <= self.config.context_length:
            raise ValueError("Expected [batch, length] tokens within the model context")
        positions = torch.arange(tokens.size(1), device=tokens.device)
        embeddings = self.token_embedding(tokens) + self.position_embedding(positions)
        return self.final_norm(self.blocks(self.dropout(embeddings)))

    def forward(self, tokens):
        return self.output(self.hidden_states(tokens))
