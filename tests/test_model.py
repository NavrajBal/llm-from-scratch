import pytest
import torch
from torch.nn import functional as F

from scratchlm.model import GELU, GPT, GPTConfig, LayerNorm


def test_normalization_and_activation_match_references():
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(LayerNorm(16)(x), F.layer_norm(x, (16,)))
    torch.testing.assert_close(GELU()(x), F.gelu(x, approximate="tanh"))


def test_gpt_shapes_gradients_and_causality():
    torch.manual_seed(3)
    model = GPT(GPTConfig(vocab_size=20, width=16, heads=2, layers=2, context_length=8))
    tokens = torch.randint(0, 20, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 20)
    changed = tokens.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 20
    torch.testing.assert_close(logits[:, :5], model(changed)[:, :5])
    F.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_context_overflow_has_clear_error():
    model = GPT(GPTConfig(context_length=4))
    with pytest.raises(ValueError, match="context"):
        model(torch.ones(1, 5, dtype=torch.long))
