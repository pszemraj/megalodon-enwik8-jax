"""Checkpoint saving and loading using Equinox serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import yaml

from .training import TrainState, create_train_state


def save_checkpoint(
    run_dir: str | Path,
    state: TrainState,
    cfg: dict[str, Any],
    tag: str | None = None,
) -> str:
    """Save checkpoint to disk.

    Saves:
    - Model parameters as .eqx file
    - Config as YAML (once per run_dir)
    - Step and optimizer state alongside model

    Args:
        run_dir: Directory to save checkpoint.
        state: Training state to save.
        cfg: Configuration dictionary.
        tag: Optional tag for checkpoint name (default: step number).

    Returns:
        Path to saved checkpoint file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Determine checkpoint name
    step = int(state.step)
    if tag:
        ckpt_name = f"checkpoint_{tag}.eqx"
    else:
        ckpt_name = f"checkpoint_{step}.eqx"

    ckpt_path = run_dir / ckpt_name

    # Save config (only once per run)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Prepare checkpoint data
    # We save as a dict with model, opt_state, step, key
    checkpoint_data = {
        "model": state.model,
        "opt_state": state.opt_state,
        "step": state.step,
        "key": state.key,
    }

    # Save using Equinox serialization
    eqx.tree_serialise_leaves(ckpt_path, checkpoint_data)

    return str(ckpt_path)


def load_checkpoint(
    ckpt_path: str | Path,
    cfg: dict[str, Any],
    key: jax.Array,
    model_builder: callable,
    optimizer: Any,
) -> TrainState:
    """Load checkpoint from disk.

    Args:
        ckpt_path: Path to checkpoint file.
        cfg: Configuration dictionary.
        key: PRNG key (may be overwritten by checkpoint).
        model_builder: Function to build model skeleton: (cfg, key) -> model.
        optimizer: Optax optimizer.

    Returns:
        Restored TrainState.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build model skeleton for deserialization
    dummy_model = model_builder(cfg, key)
    dummy_state = create_train_state(dummy_model, optimizer, key, step=0)

    # Prepare skeleton for loading
    skeleton = {
        "model": dummy_state.model,
        "opt_state": dummy_state.opt_state,
        "step": dummy_state.step,
        "key": dummy_state.key,
    }

    # Load checkpoint
    loaded = eqx.tree_deserialise_leaves(ckpt_path, skeleton)

    return TrainState(
        step=loaded["step"],
        model=loaded["model"],
        opt_state=loaded["opt_state"],
        key=loaded["key"],
    )


def load_config_from_checkpoint(ckpt_path: str | Path) -> dict[str, Any]:
    """Load config from checkpoint directory.

    Args:
        ckpt_path: Path to checkpoint file or directory.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config.yaml not found.
    """
    ckpt_path = Path(ckpt_path)

    # If path is a file, look in parent directory
    if ckpt_path.is_file():
        config_path = ckpt_path.parent / "config.yaml"
    else:
        config_path = ckpt_path / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in a run directory.

    Args:
        run_dir: Directory to search.

    Returns:
        Path to latest checkpoint, or None if no checkpoints found.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None

    checkpoints = list(run_dir.glob("checkpoint_*.eqx"))
    if not checkpoints:
        return None

    # Sort by step number (extract from filename)
    def get_step(p: Path) -> int:
        """Extract step number from checkpoint filename.

        :param Path p: Checkpoint path to parse.
        :return int: Parsed step number, or -1 if unavailable.
        """
        name = p.stem  # "checkpoint_1000"
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return -1

    checkpoints.sort(key=get_step)
    return checkpoints[-1]
