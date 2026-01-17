"""Configuration loading and validation for megalodon-enwik8-jax."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Known config keys for validation - reject unknown keys to catch typos
KNOWN_KEYS = frozenset(
    {
        # General
        "run_dir",
        "model",
        "seed",
        # Model - shared
        "num_tokens",
        "dtype",
        "jit",
        # Llama-specific
        "dim",
        "depth",
        "heads",
        "dim_head",
        "tied_embedding",
        "ffn_dim_multiplier",
        # Megalodon-specific
        "model_dim",
        "num_layers",
        "num_heads",
        "z_dim",
        "value_dim",
        "ffn_hidden_dim",
        "cema_ndim",
        "chunk_size",
        "norm_num_groups",
        "swiglu",
        "rescale_nffn",
        "scale_emb",
        "share_emb",
        "init_mode",
        "rope_base",
        "attention_dropout",
        "hidden_dropout",
        "dropout",
        "use_checkpoint",
        # Training
        "num_batches",
        "batch_size",
        "grad_accum_every",
        "learning_rate",
        "weight_decay",
        "grad_clip_norm",
        # Data
        "data_path",
        "seq_len",
        # Evaluation
        "validate_every",
        "val_batches",
        "generate_every",
        "generate_prompt_len",
        "generate_length",
        "save_every",
        "temperature",
        "min_p",
        "top_k",
        "top_p",
    }
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML config file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    return cfg


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate configuration and apply defaults.

    Args:
        cfg: Configuration dictionary to validate.

    Returns:
        Validated configuration with defaults applied.

    Raises:
        ValueError: If configuration is invalid.
    """
    # Check for unknown keys
    unknown_keys = set(cfg.keys()) - KNOWN_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {unknown_keys}")

    # Validate model type
    model = cfg.get("model", "llama").lower()
    if model not in {"megalodon", "llama"}:
        raise ValueError(f"model must be 'megalodon' or 'llama', got '{model}'")
    cfg["model"] = model

    # Validate vocab_size
    num_tokens = cfg.get("num_tokens", 256)
    if num_tokens != 256:
        raise ValueError(f"num_tokens must be 256 for enwik8 (bytes), got {num_tokens}")

    # Validate dtype
    dtype = cfg.get("dtype", "bf16").lower()
    if dtype not in {"bf16", "fp32"}:
        raise ValueError(f"dtype must be 'bf16' or 'fp32' (no fp16), got '{dtype}'")
    cfg["dtype"] = dtype

    # Megalodon-specific: chunk_size must divide seq_len
    if model == "megalodon":
        seq_len = cfg.get("seq_len", 512)
        chunk_size = cfg.get("chunk_size", seq_len)
        if seq_len > chunk_size and seq_len % chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be <= chunk_size ({chunk_size}) "
                "or divisible by it for Megalodon."
            )

    # Apply defaults for optional fields
    defaults = {
        "seed": 42,
        "jit": True,
        "batch_size": 1,
        "grad_accum_every": 1,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "validate_every": 100,
        "val_batches": 10,
        "generate_every": 100,
        "generate_prompt_len": 128,
        "generate_length": 128,
        "save_every": 500,
        "temperature": 1.0,
        "min_p": 0.1,
    }

    for key, default in defaults.items():
        if key not in cfg:
            cfg[key] = default

    return cfg


def resolve_run_dir(cfg: dict[str, Any], override: str | None = None) -> Path:
    """Resolve the run directory from config or override.

    Args:
        cfg: Configuration dictionary.
        override: Optional override path from CLI.

    Returns:
        Resolved run directory path.
    """
    if override:
        return Path(override)
    return Path(cfg.get("run_dir", "runs/default"))


def get_dtype(cfg: dict[str, Any]):
    """Get JAX dtype from config.

    Args:
        cfg: Configuration dictionary.

    Returns:
        jax.numpy dtype (bfloat16 or float32).
    """
    import jax.numpy as jnp

    dtype_str = cfg.get("dtype", "bf16").lower()
    return jnp.bfloat16 if dtype_str == "bf16" else jnp.float32
