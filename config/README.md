# Config

This folder stores the configuration for each experiment.

## Convention

- `config/experiments/`: one JSON file per experiment.
- Each experiment should record:
  - selected architecture,
  - training parameters,
  - loss configuration,
  - optimizer setup,
  - dataset path,
  - run name and tags.

## Notes

- Use one file per experiment.
- Keep the dataset path simple and explicit.
- Save the final configuration in `results/<experiment>/` after execution.
