"""Megalodon wrapper using megalodon-jax package.

Thin wrapper around megalodon-jax==0.1.0 for unified training interface.
Megalodon uses complex-valued EMA for O(n) sequence modeling with
chunk-based attention.

Key constraints:
- Requires bfloat16 or float32 (fp16 causes numerical overflow)
- chunk_size must divide seq_len (or seq_len <= chunk_size)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

# Import from megalodon-jax
try:
    from megalodon_jax import MegalodonConfig, MegalodonForCausalLM, ModelCache

    MEGALODON_JAX_AVAILABLE = True
except ImportError:
    MEGALODON_JAX_AVAILABLE = False
    MegalodonConfig = None
    MegalodonForCausalLM = None
    ModelCache = None


def assert_megalodon_available() -> None:
    """Assert megalodon-jax is available.

    Raises:
        ImportError: If megalodon-jax is not installed.
    """
    if not MEGALODON_JAX_AVAILABLE:
        raise ImportError(
            "megalodon-jax is not installed. Install with: pip install megalodon-jax==0.1.0"
        )


def assert_megalodon_version() -> None:
    """Assert megalodon-jax is the expected version.

    The plan specifies exact pin to 0.1.0 for reproducibility.

    Note: megalodon-jax doesn't expose __version__, so we just
    check that it's importable and has the expected API.
    """
    assert_megalodon_available()

    # Verify expected API exists
    required_attrs = ["MegalodonConfig", "MegalodonForCausalLM", "init_cache"]
    import megalodon_jax

    for attr in required_attrs:
        if not hasattr(megalodon_jax, attr):
            raise ImportError(
                f"megalodon-jax is missing expected attribute '{attr}'. "
                "Please install megalodon-jax==0.1.0"
            )


def build_megalodon(cfg: dict[str, Any], key: jax.Array) -> MegalodonForCausalLM:
    """Build Megalodon model from config dictionary.

    Args:
        cfg: Configuration dictionary with Megalodon parameters.
        key: PRNG key.

    Returns:
        Initialized MegalodonForCausalLM model.

    Raises:
        ImportError: If megalodon-jax is not available.
        ValueError: If config is invalid.
    """
    assert_megalodon_version()

    # Validate seq_len vs chunk_size
    seq_len = cfg.get("seq_len", 512)
    chunk_size = cfg.get("chunk_size", seq_len)
    if seq_len > chunk_size and seq_len % chunk_size != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be <= chunk_size ({chunk_size}) "
            "or divisible by it for Megalodon."
        )

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
        rope_base=cfg.get("rope_base"),
        init_mode=cfg.get("init_mode", "he"),
        use_checkpoint=cfg.get("use_checkpoint", False),
        pad_token_id=0,
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
    # megalodon-jax expects attention_mask as bool
    attention_mask = jnp.ones(input_ids.shape, dtype=bool)

    logits, new_cache = model(
        input_ids,
        attention_mask=attention_mask,
        cache=cache,
        return_cache=return_cache,
        deterministic=deterministic,
        key=key,
    )

    return logits, new_cache


def init_megalodon_cache(
    model: MegalodonForCausalLM,
    batch_size: int,
    max_seq_len: int,
) -> ModelCache:
    """Initialize cache for Megalodon generation.

    Args:
        model: MegalodonForCausalLM model.
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.

    Returns:
        Initialized ModelCache.
    """
    from megalodon_jax import init_cache

    return init_cache(model.model, batch_size, max_seq_len)


# Type alias for cache
Cache = ModelCache
