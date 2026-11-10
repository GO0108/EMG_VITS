# EMG_VITS

This repository organizes experiments and tests around the VITS architecture in a modular way, while keeping the original implementation in `vits/` as the reference base.

## Proposed structure

- `data/`: processed dataset root. Only the dataset path is registered in the experiment config.
- `model/`: architecture definitions and model factories.
- `evaluator/`: loss functions, metrics, and optimizers.
- `train/`: training and inference pipeline.
- `config/`: experiment parameters and run configuration.
- `results/`: outputs from each experiment.
- `vits/`: original VITS implementation used as the baseline.

## Goal

The goal is to isolate changes in:

1. model architecture,
2. loss functions,
3. optimizers,
4. experiment hyperparameters,
5. result tracking and comparison.

This keeps the training, evaluation, and architecture layers independent.

## Flow

1. Define an experiment in `config/experiments/`.
2. Build the model in `model/`.
3. Select losses and optimizers in `evaluator/`.
4. Run training in `train/`.
5. Save results in `results/`.

## Example

```bash
python train/run_experiment.py --config config/experiments/exp_001_vits_baseline.json
```

---

The structure is kept simple to support targeted changes and fast experiment iteration for VITS-based work.
