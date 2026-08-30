"""Classification heads and response-only instruction batches (chapters 6–7)."""

import torch
from torch import nn

from scratchlm.data import ByteTokenizer


class TextClassifier(nn.Module):
    def __init__(self, backbone, classes=2, freeze_backbone=True):
        super().__init__()
        if classes < 2:
            raise ValueError("Need at least two classes")
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        for parameter in backbone.parameters():
            parameter.requires_grad_(not freeze_backbone)
        # The LM vocabulary head is unused for classification.
        for parameter in backbone.output.parameters():
            parameter.requires_grad_(False)
        self.head = nn.Linear(backbone.config.width, classes)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, tokens, lengths):
        if lengths.shape != (tokens.size(0),):
            raise ValueError("Provide one sequence length per example")
        if ((lengths < 1) | (lengths > tokens.size(1))).any():
            raise ValueError("Invalid unpadded sequence length")
        hidden = self.backbone.hidden_states(tokens)
        rows = torch.arange(tokens.size(0), device=tokens.device)
        return self.head(hidden[rows, lengths - 1])


def classification_batch(texts, tokenizer: ByteTokenizer, context_length: int):
    if not texts or context_length < 1:
        raise ValueError("Need texts and a positive context length")
    encoded = [tokenizer.encode(text, add_eos=True)[:context_length] for text in texts]
    lengths = torch.tensor([len(ids) for ids in encoded])
    tokens = torch.full((len(encoded), int(lengths.max())), tokenizer.pad_id, dtype=torch.long)
    for row, ids in enumerate(encoded):
        tokens[row, :len(ids)] = torch.tensor(ids)
    return tokens, lengths


def instruction_prompt(record):
    prompt = f"Instruction: {record['instruction']}\n"
    if record.get("input"):
        prompt += f"Input: {record['input']}\n"
    return prompt + "Response: "


def instruction_batch(records, tokenizer: ByteTokenizer, context_length: int):
    """Right-pad and supervise only response tokens plus EOS.

    Prompt positions and padded positions use cross entropy's ignore index -100.
    Reject prompts that leave no room for a response instead of silently training
    on batches with no targets. Long responses are truncated to fit the context.
    """
    if not records or context_length < 1:
        raise ValueError("Need records and a positive context length")
    examples = []
    for record in records:
        prefix = tokenizer.encode(instruction_prompt(record))
        response = tokenizer.encode(record["output"], add_eos=True)
        if len(prefix) > context_length:
            raise ValueError("Instruction prompt leaves no room for response targets")
        full = (prefix + response)[:context_length + 1]
        x, y = full[:-1], full[1:]
        y[:len(prefix) - 1] = [-100] * (len(prefix) - 1)
        examples.append((x, y))
    length = max(len(x) for x, _ in examples)
    inputs = torch.full((len(examples), length), tokenizer.pad_id, dtype=torch.long)
    targets = torch.full_like(inputs, -100)
    for row, (x, y) in enumerate(examples):
        inputs[row, :len(x)] = torch.tensor(x)
        targets[row, :len(y)] = torch.tensor(y)
    return inputs, targets
