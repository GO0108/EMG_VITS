# Evaluator

This module groups the learning and evaluation logic.

## Suggested content

- `losses.py`: loss functions and combinations,
- `optimizers.py`: optimizer and schedule setup,
- `metrics.py`: evaluation metrics.

## Purpose

This section makes it easy to test different combinations of:

- mel reconstruction,
- KL term,
- discriminator objective,
- feature matching,
- AdamW / Adam / RMSProp,
- learning-rate tuning.

This keeps architecture changes separate from objective changes.
