import pytest
import torch

from scratchlm.data import ByteTokenizer, TokenWindows, split_tokens


def test_unicode_round_trip_and_special_tokens():
    tokenizer = ByteTokenizer()
    text = "Hello, café 🌍"
    ids = tokenizer.encode(text, add_eos=True)
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids + [tokenizer.pad_id]) == text


def test_windows_are_shifted_and_include_last_complete_window():
    windows = TokenWindows(list(range(9)), context_length=4, stride=2)
    assert len(windows) == 3
    x, y = windows[-1]
    assert torch.equal(x, torch.tensor([4, 5, 6, 7]))
    assert torch.equal(y, torch.tensor([5, 6, 7, 8]))


def test_split_does_not_share_tokens():
    train, val = split_tokens(list(range(100)), 0.8)
    assert train == list(range(80))
    assert val == list(range(80, 100))


@pytest.mark.parametrize("context,stride", [(0, 1), (2, 0), (3, 1)])
def test_bad_window_arguments(context, stride):
    with pytest.raises(ValueError):
        TokenWindows([1, 2, 3], context, stride)
