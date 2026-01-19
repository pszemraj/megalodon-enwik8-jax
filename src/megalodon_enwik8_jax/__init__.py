"""megalodon-enwik8-jax: JAX port of MEGALODON character-level LM on enwik8."""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

from .models import LlamaLM, build_model, forward_model
from .utils import (
    apply_min_p,
    apply_temperature,
    decode_tokens,
    encode_prompt,
    generate,
    load_enwik8,
    sample_accum_batch,
    sample_batch,
    sample_token,
)

__all__ = [
    "__version__",
    # Data
    "load_enwik8",
    "sample_batch",
    "sample_accum_batch",
    "encode_prompt",
    "decode_tokens",
    # Models
    "LlamaLM",
    "build_model",
    "forward_model",
    # Sampling
    "apply_temperature",
    "apply_min_p",
    "sample_token",
    "generate",
]
