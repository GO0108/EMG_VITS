from __future__ import annotations

from typing import Any, Dict

from .architectures.vits import VITSFactory


def get_model_factory(model_type: str):
    """Return a model factory for a requested type."""
    factories = {
        "vits": VITSFactory,
    }

    if model_type not in factories:
        raise ValueError(f"Unsupported model type: {model_type}")

    return factories[model_type]
