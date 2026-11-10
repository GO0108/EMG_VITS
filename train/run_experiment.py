from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VITS_ROOT = ROOT / "vits"
for entry in (str(ROOT), str(VITS_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

try:
    from evaluator.losses import build_loss_suite
    from evaluator.optimizers import build_optimizers
    from model.registry import get_model_factory
    from train.trainer import ExperimentTrainer
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", str(exc))
    if missing == "torch":
        raise SystemExit(
            "VITS dependencies are missing. Install them with: pip install -r vits/requirements.txt"
        ) from exc
    raise


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run a modular VITS experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment JSON file.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_type = config.get("architecture", {}).get("type", "vits")
    model_factory = get_model_factory(model_type)
    model = model_factory(config).build()

    loss_suite = build_loss_suite(config.get("losses", {}))
    optimizers = build_optimizers(model, model, config)
    trainer = ExperimentTrainer(model, optimizers, loss_suite, config)

    output_dir = ROOT / "results" / config.get("experiment_name", "unnamed_experiment")
    summary_path = trainer.save_summary(str(output_dir))

    print(f"Experiment: {config.get('experiment_name', 'unnamed_experiment')}")
    print(f"Model: {model_type}")
    print(f"Summary saved at: {summary_path}")


if __name__ == "__main__":
    main()
