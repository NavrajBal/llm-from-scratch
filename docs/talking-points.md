# Project talking points

Use the points that reflect work you have reviewed and can explain. Credit the
book and AI coding assistance when describing how the project was developed.

- Built an educational GPT implementation in PyTorch, inspired by Sebastian
  Raschka's *Build a Large Language Model (From Scratch)*, with AI coding assistance.
- Implemented causal multi-head attention, learned embeddings, normalization,
  residual transformer blocks, and autoregressive generation.
- Added an end-to-end pipeline for byte tokenization, next-token training, held-out
  loss evaluation, gradient clipping, and checkpoint save/load.
- Added tests against PyTorch reference attention and normalization, plus tests
  for future-token leakage, response masking, model learning, and checkpoint parity.
- Demonstrated frozen-backbone classification and response-only instruction
  fine-tuning with small synthetic datasets and explicit evaluation limits.

For a technical discussion, the recorded CPU run trained 136,832 parameters for
150 updates and reduced validation cross entropy from 5.57 to 2.57. Describe that
as a reproducible smoke test on a tiny corpus, not a benchmark or production LLM.
The project's study guide offers four weekend-sized stages; do not present it as
evidence of work completed on historical dates.
