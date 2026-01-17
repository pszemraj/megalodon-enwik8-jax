"""Logging utilities for training metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


class JsonlLogger:
    """Simple JSONL logger for training metrics.

    Appends one JSON object per line to a file, suitable for
    later analysis with pandas or other tools.
    """

    def __init__(self, path: str | Path):
        """Initialize logger.

        Args:
            path: Path to JSONL file (will be created/appended).
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any]) -> None:
        """Log a dictionary of metrics.

        JAX arrays are automatically converted to Python scalars.

        Args:
            metrics: Dictionary of metric name -> value.
        """
        # Convert JAX arrays to Python scalars
        converted = {}
        for k, v in metrics.items():
            if isinstance(v, (jax.Array, jnp.ndarray)):
                converted[k] = float(v.item()) if v.size == 1 else v.tolist()
            else:
                converted[k] = v

        with open(self.path, "a") as f:
            f.write(json.dumps(converted) + "\n")


def format_metrics(metrics: dict[str, Any], precision: int = 4) -> str:
    """Format metrics dictionary for display.

    Args:
        metrics: Dictionary of metric name -> value.
        precision: Number of decimal places for floats.

    Returns:
        Formatted string like "loss=1.2345, bpc=1.7812".
    """
    parts = []
    for k, v in metrics.items():
        if isinstance(v, (jax.Array, jnp.ndarray)):
            v = float(v.item()) if v.size == 1 else v.tolist()
        if isinstance(v, float):
            parts.append(f"{k}={v:.{precision}f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def format_step_info(
    step: int,
    total_steps: int,
    metrics: dict[str, Any],
) -> str:
    """Format step information for progress display.

    Args:
        step: Current step.
        total_steps: Total number of steps.
        metrics: Metrics dictionary.

    Returns:
        Formatted string like "[100/1200] loss=1.2345, bpc=1.7812".
    """
    metrics_str = format_metrics(metrics)
    return f"[{step}/{total_steps}] {metrics_str}"
