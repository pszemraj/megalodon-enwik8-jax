"""megalodon-enwik8-jax: JAX port of MEGALODON character-level LM on enwik8."""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

from .config import get_dtype, load_config, resolve_run_dir, validate_config
from .data import decode_tokens, encode_prompt, load_enwik8, sample_accum_batch, sample_batch
from .models import build_model, forward_model
from .training import (
    TrainState,
    bpc_from_loss,
    build_optimizer,
    create_train_state,
    cross_entropy_loss,
)

__all__ = [
    "__version__",
    # Config
    "load_config",
    "validate_config",
    "resolve_run_dir",
    "get_dtype",
    # Data
    "load_enwik8",
    "sample_batch",
    "sample_accum_batch",
    "encode_prompt",
    "decode_tokens",
    # Training
    "cross_entropy_loss",
    "bpc_from_loss",
    "build_optimizer",
    "TrainState",
    "create_train_state",
    # Models
    "build_model",
    "forward_model",
]
