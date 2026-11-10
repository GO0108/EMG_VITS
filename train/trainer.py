from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch


class ExperimentTrainer:
    """VITS-style trainer for modular experiments."""

    def __init__(self, model, optimizers, losses, config: Dict[str, Any], device: str | None = None):
        self.model = model
        self.optimizers = optimizers
        self.losses = losses
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._to_device(item) for item in value)
        return value

    def _prepare_batch(self, batch):
        if isinstance(batch, dict):
            return {key: self._to_device(value) for key, value in batch.items()}
        if isinstance(batch, (list, tuple)):
            return [self._to_device(value) for value in batch]
        return batch

    def train_step(self, batch):
        """Run one VITS-style training step with the current model and optimizer state."""
        prepared = self._prepare_batch(batch)

        if isinstance(prepared, dict):
            outputs = self.model(**prepared)
        elif isinstance(prepared, (list, tuple)):
            outputs = self.model(*prepared)
        else:
            raise TypeError("Unsupported batch format for the VITS trainer.")

        if self.optimizers is not None:
            optimizer_values = list(self.optimizers.values())
            for optimizer in optimizer_values:
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

            loss = self._compute_step_loss(outputs)
            if loss is not None:
                loss.backward()
                for optimizer in optimizer_values:
                    if optimizer is not None:
                        optimizer.step()

        return {"outputs": outputs, "status": "vits_step_complete"}

    def _compute_step_loss(self, outputs):
        """Compute a generic VITS-style loss from the model output."""
        if not isinstance(outputs, tuple):
            return None

        if not hasattr(self, "losses") or self.losses is None:
            return None

        generator_loss_fn = self.losses.get("generator_loss")
        if generator_loss_fn is None:
            return None

        if len(outputs) >= 2:
            try:
                return generator_loss_fn(outputs[0], outputs[1])
            except TypeError:
                pass

        if len(outputs) >= 3:
            try:
                return generator_loss_fn(outputs[0], outputs[2])
            except TypeError:
                pass

        return None

    def validate(self, batch):
        """Run a validation-style forward pass."""
        step = self.train_step(batch)
        return {"status": "ok", "metrics": step.get("outputs", {})}

    def run(self, train_loader=None, eval_loader=None, epochs: int | None = None):
        """Run a simple VITS-like training cycle."""
        epochs = epochs or int(self.config.get("training", {}).get("epochs", 1))
        history: List[Dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            epoch_metrics = {"epoch": epoch, "train": [], "eval": []}

            if train_loader is not None:
                for batch in train_loader:
                    epoch_metrics["train"].append(self.train_step(batch))

            if eval_loader is not None:
                for batch in eval_loader:
                    epoch_metrics["eval"].append(self.validate(batch))

            history.append(epoch_metrics)

        return history

    def save_summary(self, output_dir: str):
        """Save the experiment summary to disk."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        summary = {
            "experiment": self.config.get("experiment_name", "unknown"),
            "architecture": self.config.get("architecture", {}),
            "training": self.config.get("training", {}),
            "device": self.device,
        }
        summary_path = path / "experiment_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return str(summary_path)
