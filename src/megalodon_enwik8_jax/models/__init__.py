"""Model implementations for megalodon-enwik8-jax.

Provides unified interface for both Llama and Megalodon models.
"""

from __future__ import annotations

from typing import Any

import jax
from jaxtyping import Array, Float, Int

from .llama import LlamaLM, build_llama, forward_llama
from .megalodon import MEGALODON_JAX_AVAILABLE, build_megalodon, forward_megalodon


def build_model(cfg: dict[str, Any], key: jax.Array):
    """Build model based on config.

    Args:
        cfg: Configuration dictionary with "model" key.
        key: PRNG key.

    Returns:
        Initialized model (LlamaLM or MegalodonForCausalLM).
    """
    model_type = cfg.get("model", "llama").lower()
    if model_type == "llama":
        return build_llama(cfg, key)
    elif model_type == "megalodon":
        if not MEGALODON_JAX_AVAILABLE:
            raise ImportError(
                "megalodon-jax required. Install with: pip install megalodon-jax==0.1.0"
            )
        return build_megalodon(cfg, key)
    raise ValueError(f"Unknown model type: {model_type}")


def forward_model(
    model,
    input_ids: Int[Array, "batch seq"],
    cache=None,
    return_cache: bool = False,
    deterministic: bool = True,
    key: jax.Array | None = None,
) -> tuple[Float[Array, "batch seq vocab"], Any]:
    """Forward pass for any model.

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
        return forward_llama(model, input_ids, cache, return_cache, deterministic, key)
    return forward_megalodon(model, input_ids, cache, return_cache, deterministic, key)


__all__ = [
    "build_model",
    "forward_model",
    "LlamaLM",
    "build_llama",
    "forward_llama",
    "build_megalodon",
    "forward_megalodon",
]
