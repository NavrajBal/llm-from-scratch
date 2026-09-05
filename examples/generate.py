"""Generate text from a local scratchlm checkpoint."""

import argparse

import torch

from scratchlm.data import ByteTokenizer
from scratchlm.generation import generate
from scratchlm.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/tiny/model.pt")
    parser.add_argument("--prompt", default="The robot")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    model = load_checkpoint(args.checkpoint)
    tokenizer = ByteTokenizer()
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    result = generate(model, prompt, args.tokens, args.temperature, args.top_k, tokenizer.eos_id)
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
