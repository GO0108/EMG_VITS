from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def load_checkpoint(model, checkpoint_path: str, map_location: str | None = None):
    """Load a checkpoint into a VITS model."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(str(path), map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if hasattr(model, "load_state_dict"):
        model.load_state_dict(state)
    return model


def synthesize(model, text: str, **kwargs: Any) -> Dict[str, Any]:
    """Prepare a VITS-style inference call. The model should implement infer()."""
    payload = {
        "text": text,
        "model": model.__class__.__name__,
        "kwargs": kwargs,
        "status": "ready_for_inference",
    }

    if hasattr(model, "infer"):
        payload["call"] = "model.infer"
    return payload
