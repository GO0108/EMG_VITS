# Train

This folder holds the training and inference pipeline.

## Layout

- `trainer.py`: training loop and orchestration,
- `inference.py`: synthesis and inference helpers,
- `run_experiment.py`: entry point for experiment execution.

## Scope

- load the experiment config,
- build the model,
- load the dataset,
- run training,
- save checkpoints,
- log outputs in `results/`.
