import pytest
import torch

from scratchlm.data import ByteTokenizer
from scratchlm.finetuning import (
    TextClassifier, classification_batch, instruction_batch, instruction_prompt,
)
from scratchlm.model import GPT, GPTConfig
from scratchlm.training import language_loss


def test_classifier_ignores_right_padding_and_freezes_backbone():
    tokenizer = ByteTokenizer()
    backbone = GPT(GPTConfig(width=16, heads=2, layers=1))
    model = TextClassifier(backbone).train()
    x, lengths = classification_batch(["yes", "a longer message"], tokenizer, 64)
    logits = model(x, lengths)
    single = model(x[:1, :lengths[0]], lengths[:1])
    torch.testing.assert_close(logits[:1], single)
    logits.sum().backward()
    assert model.head.weight.grad is not None
    assert all(p.grad is None for p in backbone.parameters())
    assert not backbone.training


def test_instruction_targets_cover_only_response_and_eos():
    tokenizer = ByteTokenizer()
    records = [
        {"instruction": "Say hi", "output": "hello"},
        {"instruction": "Repeat", "input": "a", "output": "a"},
    ]
    x, y = instruction_batch(records, tokenizer, 80)
    for row, record in enumerate(records):
        prefix_length = len(tokenizer.encode(instruction_prompt(record)))
        assert (y[row, :prefix_length - 1] == -100).all()
        assert y[row][y[row] != -100].tolist() == tokenizer.encode(record["output"], add_eos=True)
    logits = torch.randn(*x.shape, tokenizer.vocab_size, requires_grad=True)
    language_loss(logits, y).backward()
    assert (logits.grad[y == -100] == 0).all()
    assert (logits.grad[y != -100].abs().sum(dim=-1) > 0).all()


def test_long_prompt_is_rejected():
    with pytest.raises(ValueError, match="no room"):
        instruction_batch([{"instruction": "long prompt", "output": "ok"}], ByteTokenizer(), 8)
