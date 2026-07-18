"""Thin unified-interface adapter for megalodon-jax 0.2."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from megalodon_jax import MegalodonConfig, MegalodonForCausalLM, ModelCache, init_cache


def _resolve_dtype(value: Any, default: jnp.dtype) -> jnp.dtype:
    """Resolve a JAX data type from a configuration value.

    Args:
        value: Configured dtype name or JAX dtype, or ``None``.
        default: Data type to return when ``value`` is ``None``.

    Returns:
        The resolved JAX data type.
    """
    if value is None:
        return default
    if isinstance(value, str):
        val = value.lower()
        if val in {"bf16", "bfloat16"}:
            return jnp.bfloat16
        if val in {"fp32", "float32"}:
            return jnp.float32
        raise ValueError(f"Precision policy dtype must be bf16/fp32, got '{value}'")
    if isinstance(value, jnp.dtype):
        return value
    if value in {jnp.bfloat16, jnp.float32}:
        return jnp.dtype(value)
    raise ValueError(f"Precision policy dtype must be bf16/fp32, got '{value}'")


def build_megalodon(cfg: dict[str, Any], key: jax.Array) -> MegalodonForCausalLM:
    """Build Megalodon model from config dictionary.

    Args:
        cfg: Configuration dictionary with Megalodon parameters.
        key: PRNG key.

    Returns:
        Initialized MegalodonForCausalLM model.

    Raises:
        ValueError: If config is invalid.
    """
    chunk_size = cfg.get("chunk_size", cfg.get("seq_len", 512))
    compute_dtype = _resolve_dtype(cfg.get("compute_dtype"), jnp.bfloat16)
    param_dtype = _resolve_dtype(cfg.get("param_dtype"), jnp.float32)
    accum_dtype = _resolve_dtype(cfg.get("accum_dtype"), jnp.float32)
    attention_softmax_dtype = _resolve_dtype(cfg.get("attention_softmax_dtype"), jnp.float32)
    loss_softmax_dtype = _resolve_dtype(cfg.get("loss_softmax_dtype"), jnp.float32)

    # Build config with mapped parameters
    config = MegalodonConfig(
        vocab_size=cfg.get("num_tokens", 256),
        model_dim=cfg.get("model_dim", 384),
        num_layers=cfg.get("num_layers", 6),
        num_heads=cfg.get("num_heads", 3),
        z_dim=cfg.get("z_dim", 192),
        value_dim=cfg.get("value_dim", 384),
        ffn_hidden_dim=cfg.get("ffn_hidden_dim", 1024),
        cema_ndim=cfg.get("cema_ndim", 8),
        chunk_size=chunk_size,
        norm_num_groups=cfg.get("norm_num_groups", 32),
        dropout=cfg.get("dropout", 0.0),
        attention_dropout=cfg.get("attention_dropout", 0.0),
        hidden_dropout=cfg.get("hidden_dropout", 0.0),
        swiglu=cfg.get("swiglu", True),
        rescale_nffn=cfg.get("rescale_nffn", False),
        scale_emb=cfg.get("scale_emb", False),
        share_emb=cfg.get("share_emb", False),
        rope_base=cfg.get("rope_base"),
        init_mode=cfg.get("init_mode", "he"),
        use_checkpoint=cfg.get("use_checkpoint", False),
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
        attention_softmax_dtype=attention_softmax_dtype,
        loss_softmax_dtype=loss_softmax_dtype,
        pad_token_id=None,
    )

    return MegalodonForCausalLM(config, key=key)


def forward_megalodon(
    model: MegalodonForCausalLM,
    input_ids: Int[Array, "batch seq"],
    cache: ModelCache | None = None,
    return_cache: bool = False,
    deterministic: bool = True,
    key: jax.Array | None = None,
) -> tuple[Float[Array, "batch seq vocab"], ModelCache | None]:
    """Forward pass for Megalodon model.

    Args:
        model: MegalodonForCausalLM model.
        input_ids: Input token IDs of shape [B, T].
        cache: Optional model cache for generation.
        return_cache: Whether to return updated cache.
        deterministic: Whether to use deterministic mode.
        key: PRNG key for dropout (if not deterministic).

    Returns:
        Tuple of (logits, cache) where:
        - logits: Shape [B, T, vocab_size]
        - cache: Updated cache (or None if not requested)
    """
    # attention_mask defaults to None in megalodon-jax (no masking for unpadded seqs)
    logits, new_cache = model(
        input_ids,
        cache=cache,
        return_cache=return_cache,
        deterministic=deterministic,
        key=key,
    )

    return logits, new_cache


def init_megalodon_cache(
    model: MegalodonForCausalLM,
) -> ModelCache:
    """Initialize the sparse Megalodon continuation cache.

    Args:
        model: MegalodonForCausalLM model.

    Returns:
        Initialized ModelCache.
    """
    return init_cache(model.config)


# Type alias for cache
Cache = ModelCache
