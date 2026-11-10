# Model

This folder holds VITS-based architecture changes.

## Responsibility

- define the main model architecture,
- encapsulate block and network variations,
- expose a model factory per experiment,
- keep architecture code separate from training code.

## Suggested structure

- `architectures/`: concrete implementations,
- `registry.py`: factory selection by config,
- `__init__.py`: exports.

## Typical changes

- hidden-channel tuning,
- layer count changes,
- conditional block additions,
- encoder/decoder swaps,
- duration predictor changes.
