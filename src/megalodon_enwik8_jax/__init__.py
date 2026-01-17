"""megalodon-enwik8-jax: JAX port of MEGALODON character-level LM on enwik8."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import get_dtype, load_config, resolve_run_dir, validate_config
from .data import decode_tokens, encode_prompt, load_enwik8, sample_accum_batch, sample_batch
from .losses import bpc_from_loss, cross_entropy_loss
from .models import build_model, forward_model, get_model_type
from .optim import build_optimizer
from .train_state import TrainState, create_train_state
from .utils import count_params, format_params, model_summary

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
    # Losses
    "cross_entropy_loss",
    "bpc_from_loss",
    # Models
    "build_model",
    "forward_model",
    "get_model_type",
    # Optimizer
    "build_optimizer",
    # Training
    "TrainState",
    "create_train_state",
    # Utils
    "count_params",
    "format_params",
    "model_summary",
]
