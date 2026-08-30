import torch
from torch.utils.data import DataLoader

from scratchlm.data import TokenWindows
from scratchlm.model import GPT, GPTConfig
from scratchlm.training import evaluate, load_checkpoint, save_checkpoint, train_steps


def test_model_learns_and_checkpoint_preserves_predictions(tmp_path):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = GPT(GPTConfig(vocab_size=4, context_length=8, width=16, heads=2, layers=1))
    loader = DataLoader(TokenWindows([0, 1, 2, 3] * 32, 8), batch_size=4)
    before = evaluate(model, loader)
    losses = train_steps(model, loader, steps=30, learning_rate=0.01)
    after = evaluate(model, loader)
    assert after < before * 0.25
    assert len(losses) == 30
    assert model.training  # evaluate restores the mode.
    path = tmp_path / "model.pt"
    save_checkpoint(model, path)
    restored = load_checkpoint(path)
    model.eval()
    x, _ = next(iter(loader))
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
