"""Pytest configuration and fixtures for megalodon-enwik8-jax tests."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Numpy random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def key() -> jax.Array:
    """JAX PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def test_config() -> dict[str, Any]:
    """Minimal config for testing."""
    return {
        "model": "llama",
        "num_tokens": 256,
        "dtype": "bf16",
        "dim": 64,
        "depth": 2,
        "heads": 2,
        "dim_head": 32,
        "ffn_dim_multiplier": 2.0,
        "tied_embedding": True,
        "seq_len": 32,
        "batch_size": 2,
        "grad_accum_every": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "seed": 42,
        "data_path": "data/enwik8.gz",
        "validate_every": 100,
        "val_batches": 2,
        "generate_every": 100,
        "generate_prompt_len": 16,
        "generate_length": 16,
        "save_every": 500,
        "temperature": 1.0,
        "min_p": 0.1,
    }


@pytest.fixture
def megalodon_config() -> dict[str, Any]:
    """Minimal Megalodon config for testing."""
    return {
        "model": "megalodon",
        "num_tokens": 256,
        "dtype": "bf16",
        "model_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "z_dim": 16,
        "value_dim": 32,
        "ffn_hidden_dim": 128,
        "cema_ndim": 8,
        "chunk_size": 32,
        "norm_num_groups": 1,
        "seq_len": 32,
        "batch_size": 2,
        "grad_accum_every": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "seed": 42,
        "data_path": "data/enwik8.gz",
        "validate_every": 100,
        "val_batches": 2,
        "generate_every": 100,
        "generate_prompt_len": 16,
        "generate_length": 16,
        "save_every": 500,
        "temperature": 1.0,
        "min_p": 0.1,
    }
