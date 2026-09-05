"""Fit a classification head on frozen language-model features.

Eight synthetic messages are a training smoke test, not a spam benchmark.
"""

import argparse
import json

import torch
from torch.nn import functional as F

from scratchlm.data import ByteTokenizer
from scratchlm.finetuning import TextClassifier, classification_batch
from scratchlm.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/tiny/model.pt")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    backbone = load_checkpoint(args.checkpoint)
    model = TextClassifier(backbone).train()
    texts = [
        "Win a free prize now", "Claim your cash reward", "Click for a free gift",
        "You won a bonus prize", "Meet me at the library", "Dinner is ready",
        "Can you call me later", "The meeting starts at noon",
    ]
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    tokens, lengths = classification_batch(texts, ByteTokenizer(), backbone.config.context_length)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.03)
    with torch.no_grad():
        initial_loss = F.cross_entropy(model(tokens, lengths), labels).item()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(tokens, lengths), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(tokens, lengths)
        print(json.dumps({
            "examples": len(texts), "steps": args.steps,
            "initial_training_loss": initial_loss,
            "final_training_loss": F.cross_entropy(logits, labels).item(),
            "training_accuracy": (logits.argmax(-1) == labels).float().mean().item(),
            "note": "Training-set smoke test only; no held-out accuracy claim.",
        }, indent=2))


if __name__ == "__main__":
    main()
