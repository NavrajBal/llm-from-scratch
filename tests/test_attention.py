import pytest
import torch
from torch.nn import functional as F

from scratchlm.attention import CausalAttention


def test_attention_matches_pytorch_reference():
    torch.manual_seed(7)
    attention = CausalAttention(12, 3, 8).eval()
    x = torch.randn(2, 5, 12)
    q, k, v = attention.qkv(x).view(2, 5, 3, 3, 4).permute(2, 0, 3, 1, 4).unbind(0)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = attention.projection(expected.transpose(1, 2).reshape(2, 5, 12))
    torch.testing.assert_close(attention(x), expected)


def test_future_tokens_cannot_change_past_outputs():
    torch.manual_seed(11)
    attention = CausalAttention(12, 3, 8).eval()
    x = torch.randn(2, 8, 12)
    modified = x.clone()
    modified[:, 4:] = torch.randn_like(modified[:, 4:]) * 100
    torch.testing.assert_close(attention(x)[:, :4], attention(modified)[:, :4])


def test_invalid_head_count():
    with pytest.raises(ValueError):
        CausalAttention(10, 3, 8)
