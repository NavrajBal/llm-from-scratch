"""Autoregressive greedy or temperature/top-k sampling."""

import math

import torch


@torch.no_grad()
def generate(model, tokens, max_new_tokens=50, temperature=0.0, top_k=None, eos_id=None):
    if tokens.ndim != 2 or tokens.size(0) != 1 or tokens.size(1) == 0:
        raise ValueError("Provide one nonempty prompt with shape [1, length]")
    if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Token count and finite temperature must be nonnegative")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = tokens[:, -model.config.context_length:]
            logits = model(context)[:, -1, :]
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    filtered = torch.full_like(logits, float("-inf"))
                    logits = filtered.scatter(-1, indices, values)
                next_token = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
            if eos_id is not None and next_token.item() == eos_id:
                break
    finally:
        model.train(was_training)
    return tokens
