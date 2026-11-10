from __future__ import annotations

from typing import Dict, List


def compute_training_metrics(losses: Dict[str, float], step: int) -> Dict[str, float]:
    """Return a simple metric dict for logging."""
    metrics = {"step": step}
    for key, value in losses.items():
        metrics[f"loss/{key}"] = float(value)
    return metrics


def summarize_metrics(history: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute basic aggregates for a metric history."""
    if not history:
        return {}

    keys = history[0].keys()
    summary: Dict[str, float] = {}
    for key in keys:
        values = [entry.get(key, 0.0) for entry in history]
        summary[f"avg_{key}"] = sum(values) / len(values)
    return summary
