"""Model implementations for megalodon-enwik8-jax.

Provides unified interface for both Llama and Megalodon models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
from jaxtyping import Array, Float, Int

from .llama import LlamaLM, build_llama, forward_llama
from .megalodon import MegalodonForCausalLM, build_megalodon, forward_megalodon

if TYPE_CHECKING:
    from .megalodon import ModelCache

LlamaCache = list[tuple[Array, Array]]


def build_model(cfg: dict[str, Any], key: jax.Array) -> LlamaLM | MegalodonForCausalLM:
    """Build model based on config.

    Args:
        cfg: Configuration dictionary with a ``model`` key.
        key: PRNG key.

    Returns:
        Initialized model instance.

    Raises:
        ValueError: If the configured model type is unknown.
    """
    model_type = cfg.get("model", "llama").lower()
    if model_type == "llama":
        model = build_llama(cfg, key)
    elif model_type == "megalodon":
        model = build_megalodon(cfg, key)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def forward_model(
    model: LlamaLM | MegalodonForCausalLM,
    input_ids: Int[Array, "batch seq"],
    cache: LlamaCache | ModelCache | None = None,
    return_cache: bool = False,
    deterministic: bool = True,
    key: jax.Array | None = None,
) -> tuple[Float[Array, "batch seq vocab"], LlamaCache | ModelCache | None]:
    """Forward pass for any model.

    Args:
        model: Supported model instance.
        input_ids: Input token IDs of shape ``[batch, sequence]``.
        cache: Optional cache for generation.
        return_cache: Whether to return the updated cache.
        deterministic: Whether to use deterministic mode.
        key: PRNG key for stochastic layers.

    Returns:
        Logits and the optional updated cache.

    Raises:
        TypeError: If ``model`` is not a supported model instance.
    """
    if isinstance(model, LlamaLM):
        return forward_llama(model, input_ids, cache, return_cache, deterministic, key)
    if isinstance(model, MegalodonForCausalLM):
        return forward_megalodon(model, input_ids, cache, return_cache, deterministic, key)
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


__all__ = [
    "build_model",
    "forward_model",
    "LlamaLM",
    "build_llama",
    "forward_llama",
    "build_megalodon",
    "forward_megalodon",
]
