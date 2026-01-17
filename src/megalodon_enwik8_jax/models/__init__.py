"""Model implementations for megalodon-enwik8-jax.

Provides unified interface for both Llama and Megalodon models.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
from jaxtyping import Array, Float, Int

from .llama import LlamaLM, build_llama, forward_llama
from .megalodon import (
    MEGALODON_JAX_AVAILABLE,
    build_megalodon,
    forward_megalodon,
)

# Type aliases
Model = eqx.Module
Cache = Any  # Opaque cache type (model-specific)


def build_model(cfg: dict[str, Any], key: jax.Array) -> Model:
    """Build model based on config.

    Dispatches to build_llama or build_megalodon based on cfg["model"].

    Args:
        cfg: Configuration dictionary.
        key: PRNG key.

    Returns:
        Initialized model (LlamaLM or MegalodonForCausalLM).

    Raises:
        ValueError: If model type is unknown.
        ImportError: If megalodon-jax not available for megalodon model.
    """
    model_type = cfg.get("model", "llama").lower()

    if model_type == "llama":
        return build_llama(cfg, key)
    elif model_type == "megalodon":
        if not MEGALODON_JAX_AVAILABLE:
            raise ImportError(
                "megalodon-jax is required for megalodon model. "
                "Install with: pip install megalodon-jax==0.1.0"
            )
        return build_megalodon(cfg, key)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def forward_model(
    model: Model,
    input_ids: Int[Array, "batch seq"],
    cache: Cache | None = None,
    return_cache: bool = False,
    deterministic: bool = True,
    key: jax.Array | None = None,
) -> tuple[Float[Array, "batch seq vocab"], Cache | None]:
    """Forward pass for any model.

    Dispatches to forward_llama or forward_megalodon based on model type.

    Args:
        model: Model instance.
        input_ids: Input token IDs of shape [B, T].
        cache: Optional cache for generation.
        return_cache: Whether to return updated cache.
        deterministic: Whether to use deterministic mode.
        key: PRNG key.

    Returns:
        Tuple of (logits, cache).
    """
    if isinstance(model, LlamaLM):
        return forward_llama(
            model,
            input_ids,
            cache=cache,
            return_cache=return_cache,
            deterministic=deterministic,
            key=key,
        )
    else:
        # Assume megalodon
        return forward_megalodon(
            model,
            input_ids,
            cache=cache,
            return_cache=return_cache,
            deterministic=deterministic,
            key=key,
        )


def get_model_type(model: Model) -> str:
    """Get the type of model as a string.

    Args:
        model: Model instance.

    Returns:
        "llama" or "megalodon".
    """
    if isinstance(model, LlamaLM):
        return "llama"
    return "megalodon"


__all__ = [
    "Model",
    "Cache",
    "build_model",
    "forward_model",
    "get_model_type",
    "LlamaLM",
    "build_llama",
    "forward_llama",
    "build_megalodon",
    "forward_megalodon",
]
