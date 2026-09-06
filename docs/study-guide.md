# Four-weekend study guide

This is a suggested learning sequence, not a record of past work dates. Each stage
has a working implementation and an experiment to try while reading the book.

## Weekend 1: text to tokens and attention

Read the chapter 1 overview and chapter 2 data pipeline concepts. Inspect
`src/scratchlm/data.py` and `tests/test_data.py`. Print a training window and verify
that each target is the following token. Try multilingual text with the tokenizer.
Compare byte tokenization with the book's BPE approach: bytes keep the vocabulary
small but usually produce longer sequences.

Then read chapter 3 and inspect `attention.py`. Follow the tensor shapes through
Q/K/V projections and multiple heads. Run the future-token perturbation test.
As a local experiment, remove the causal mask and observe that the test fails.
Restore the mask before committing any changes.

## Weekend 2: assemble the decoder

Read chapter 4 and inspect `model.py`. Trace embeddings, normalization, attention,
residual connections, the feed-forward layer, and the vocabulary head. Explain
why positional embeddings are needed and why normalization uses population
variance. Compare the hand-written components to their PyTorch references.

Inspect `generation.py`. Compare greedy decoding with temperature and top-k.
The model generates one token at a time, appends it, and repeats. Cropping the
context bounds the input size but does not provide a KV cache.

## Weekend 3: train, evaluate, and save

Read chapter 5 and inspect `training.py` and `examples/train.py`. Run the default
150-step experiment, then change one setting, such as model width or step count.
Keep the seed and corpus fixed so the comparison is meaningful. Use your own
appropriately licensed text with `--text path/to/corpus.txt` for a larger experiment.

Explain why the split happens before sliding windows and why training loss alone
is insufficient. The included validation tail is much too small for strong claims.
Save and reload a checkpoint and compare predictions. This checkpoint format is
for inference and transfer learning; exact optimizer resumption is out of scope.

## Weekend 4: adapt the model

Read chapters 6–7 and inspect `finetuning.py`. Run `examples/classify.py` after
training a checkpoint. The classifier pools the last real token, so appending
right padding does not change its features. Only the linear classification head
is optimized by the demo; the language-model backbone is frozen.

Run `examples/instruct.py`. Identify which targets receive loss: response tokens
and EOS, with prompt and padding targets ignored. Padding is on the right, so
causal attention prevents padded future positions from affecting real tokens.
The fine-tuning examples demonstrate mechanics on training data only. A next
experiment would need a separate held-out set, a documented data source, and
evaluation beyond memorizing these examples.

## Questions to be able to answer

- What breaks if next-token inputs and targets are not shifted?
- Why divide attention scores by the square root of head dimension?
- How does causal masking prevent future information from leaking?
- What do residual connections and pre-normalization contribute?
- Why can a lower training loss coexist with worse validation loss?
- How do classification and instruction tuning change the learning objective?
- Which claims are supported by these smoke tests, and which need larger studies?
