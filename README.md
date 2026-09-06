# LLM from scratch

A small PyTorch implementation of GPT building blocks, inspired by Sebastian
Raschka's [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch).

## Features

- Byte tokenization and next-token data preparation
- Causal multi-head attention and transformer blocks
- Training, checkpointing, and text generation
- Classification and instruction fine-tuning examples

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Usage

Run from the repository root. Train first to create the checkpoint used by the
other examples.

```bash
python examples/train.py
python examples/generate.py --prompt "The robot"
python examples/classify.py
python examples/instruct.py
python -m pytest -q
```

Examples use small datasets and run on CPU by default. Outputs are saved in `runs/`.
See the [study guide](docs/study-guide.md) for explanations and experiments.

Educational project developed with AI assistance. Commit dates represent a
retrospective study sequence.

[MIT License](LICENSE)
