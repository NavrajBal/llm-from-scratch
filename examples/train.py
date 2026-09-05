"""Train a small GPT on local text and record an auditable smoke-test run."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scratchlm.data import ByteTokenizer, TokenWindows, split_tokens
from scratchlm.generation import generate
from scratchlm.model import GPT, GPTConfig
from scratchlm.training import evaluate, save_checkpoint, train_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, default=Path("data/tiny_stories.txt"))
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/tiny"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode(args.text.read_text(encoding="utf-8"))
    train_tokens, val_tokens = split_tokens(tokens)
    config = GPTConfig()
    train_data = TokenWindows(train_tokens, config.context_length, stride=16)
    val_data = TokenWindows(val_tokens, config.context_length)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)
    model = GPT(config).to(args.device)
    initial_val = evaluate(model, val_loader)
    losses = train_steps(model, train_loader, args.steps)
    metrics = {
        "seed": args.seed,
        "steps": args.steps,
        "device": args.device,
        "torch_version": str(torch.__version__),
        "config": asdict(config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_windows": len(train_data),
        "validation_windows": len(val_data),
        "initial_validation_loss": initial_val,
        "final_training_loss": evaluate(model, train_loader),
        "final_validation_loss": evaluate(model, val_loader),
        "step_losses": losses,
    }
    prompt = torch.tensor([tokenizer.encode("The robot")], device=args.device)
    metrics["greedy_sample"] = tokenizer.decode(generate(model, prompt, 100)[0].tolist())
    args.output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, args.output / "model.pt")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "step_losses"}, indent=2))


if __name__ == "__main__":
    main()
