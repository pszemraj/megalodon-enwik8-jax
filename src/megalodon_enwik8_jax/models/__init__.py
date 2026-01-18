"""Model implementations for megalodon-enwik8-jax.

Provides unified interface for both Llama and Megalodon models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
from jaxtyping import Array, Float, Int

from .llama import LlamaLM, build_llama, forward_llama
from .megalodon import MEGALODON_JAX_AVAILABLE, build_megalodon, forward_megalodon

if TYPE_CHECKING:
    from .megalodon import MegalodonForCausalLM, ModelCache

LlamaCache = list[tuple[Array, Array]]


def build_model(cfg: dict[str, Any], key: jax.Array) -> LlamaLM | MegalodonForCausalLM:
    """Build model based on config.

    :param dict[str, Any] cfg: Configuration dictionary with "model" key.
    :param jax.Array key: PRNG key.
    :raises ImportError: If Megalodon is requested but unavailable.
    :raises ValueError: If model type is unknown.
    :return LlamaLM | MegalodonForCausalLM: Initialized model instance.
    """
    model_type = cfg.get("model", "llama").lower()
    if model_type == "llama":
        model = build_llama(cfg, key)
    elif model_type == "megalodon":
        if not MEGALODON_JAX_AVAILABLE:
            raise ImportError(
                "megalodon-jax required. Install with: pip install megalodon-jax==0.1.0"
            )
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

    :param LlamaLM | MegalodonForCausalLM model: Model instance.
    :param Int[Array, "batch seq"] input_ids: Input token IDs of shape [B, T].
    :param LlamaCache | ModelCache | None cache: Optional cache for generation.
    :param bool return_cache: Whether to return updated cache.
    :param bool deterministic: Whether to use deterministic mode.
    :param jax.Array | None key: PRNG key.
    :return tuple[Float[Array, "batch seq vocab"], LlamaCache | ModelCache | None]:
        Logits and cache tuple.
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
