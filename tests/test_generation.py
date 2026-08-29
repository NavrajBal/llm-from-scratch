import pytest
import torch

from scratchlm.generation import generate
from scratchlm.model import GPT, GPTConfig


def test_generation_crops_context_and_restores_mode():
    model = GPT(GPTConfig(vocab_size=8, width=8, heads=2, layers=1, context_length=4))
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    greedy = generate(model, prompt, max_new_tokens=5)
    top_one = generate(model, prompt, max_new_tokens=5, temperature=1, top_k=1)
    assert greedy.shape == (1, 10)
    assert model.training
    torch.testing.assert_close(greedy, top_one)
    torch.testing.assert_close(greedy[:, :5], prompt)


def test_eos_stops_and_invalid_top_k_fails():
    model = GPT(GPTConfig(vocab_size=8, width=8, heads=2, layers=1))
    with torch.no_grad():
        model.output.weight.zero_()  # Greedy tie selects ID 0.
    prompt = torch.tensor([[1]])
    assert generate(model, prompt, eos_id=0).tolist() == [[1, 0]]
    with pytest.raises(ValueError, match="top_k"):
        generate(model, prompt, top_k=0)
