"""Next-token optimization, evaluation, and model checkpoints (chapter 5)."""

from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from scratchlm.model import GPT, GPTConfig


def language_loss(logits, targets):
    if not (targets != -100).any():
        raise ValueError("A batch must contain at least one supervised target")
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)


@torch.no_grad()
def evaluate(model, loader):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_tokens = 0.0, 0
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            count = (y != -100).sum().item()
            if count:
                total_loss += language_loss(model(x), y).item() * count
                total_tokens += count
    finally:
        model.train(was_training)
    if not total_tokens:
        raise ValueError("Evaluation requires supervised tokens")
    return total_loss / total_tokens


def train_steps(model, loader, steps: int, learning_rate=3e-3):
    if steps < 1 or learning_rate <= 0 or len(loader) == 0:
        raise ValueError("Need positive steps, learning rate, and a nonempty loader")
    model.train()
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    iterator = iter(loader)
    losses = []
    for _ in range(steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = language_loss(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is not finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
    return losses


def save_checkpoint(model: GPT, path):
    """Save inference/transfer weights, not optimizer state for exact resumption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "config": asdict(model.config),
                "model": model.state_dict()}, path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()
