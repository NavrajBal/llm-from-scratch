"""Exercise response-only instruction tuning on four synthetic records."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from scratchlm.data import ByteTokenizer
from scratchlm.finetuning import instruction_batch, instruction_prompt
from scratchlm.generation import generate
from scratchlm.training import evaluate, load_checkpoint, save_checkpoint, train_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/tiny/model.pt")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("runs/instruction/model.pt"))
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = load_checkpoint(args.checkpoint)
    tokenizer = ByteTokenizer()
    records = [
        {"instruction": "Say hello", "output": "Hello!"},
        {"instruction": "Count to three", "output": "One, two, three."},
        {"instruction": "Name a color", "output": "Blue."},
        {"instruction": "Say goodbye", "output": "Goodbye!"},
    ]
    x, y = instruction_batch(records, tokenizer, model.config.context_length)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    before = evaluate(model, loader)
    train_steps(model, loader, args.steps, learning_rate=1e-3)
    after = evaluate(model, loader)
    save_checkpoint(model, args.output)
    prompt = torch.tensor([tokenizer.encode(instruction_prompt(records[0]))])
    result = generate(model, prompt, 20, eos_id=tokenizer.eos_id)
    print(json.dumps({
        "examples": len(records), "steps": args.steps,
        "initial_training_loss": before, "final_training_loss": after,
        "training_prompt_sample": tokenizer.decode(result[0].tolist()),
        "note": "Training-set smoke test only; not a general instruction-following model.",
    }, indent=2))


if __name__ == "__main__":
    main()
