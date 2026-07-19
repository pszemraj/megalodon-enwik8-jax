"""Llama-style transformer in Equinox.

Standard autoregressive transformer with:
- RMSNorm (Root Mean Square Layer Normalization)
- SwiGLU feedforward
- RoPE (Rotary Position Embeddings)
- Pre-normalization architecture

References:
- Llama paper: https://arxiv.org/abs/2302.13971
- Llama-2 paper: https://arxiv.org/abs/2307.09288
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def _linear_3d(
    linear: eqx.nn.Linear,
    x: Array,
    compute_dtype: jnp.dtype,
) -> Array:
    """Apply a linear layer over a batched sequence.

    Args:
        linear: Linear projection to apply.
        x: Input array with shape ``[batch, sequence, input_dim]``.
        compute_dtype: Data type used for the projection computation.

    Returns:
        Projected array with shape ``[batch, sequence, output_dim]``.
    """
    x_cast = x.astype(compute_dtype)
    weight = linear.weight.astype(compute_dtype)
    y = jnp.einsum("btd,od->bto", x_cast, weight, preferred_element_type=jnp.float32)
    if linear.bias is not None:
        y = y + linear.bias.astype(compute_dtype)
    return y.astype(compute_dtype)


def _resolve_compute_dtype(cfg: dict[str, Any]) -> jnp.dtype:
    """Resolve the configured compute data type.

    Args:
        cfg: Model configuration containing ``compute_dtype`` or the legacy
            ``dtype`` key.

    Returns:
        The corresponding JAX data type.
    """
    value = cfg.get("compute_dtype", cfg.get("dtype", "bf16"))
    if isinstance(value, str):
        val = value.lower()
        if val in {"bf16", "bfloat16"}:
            return jnp.bfloat16
        if val in {"fp32", "float32"}:
            return jnp.float32
        raise ValueError(f"compute_dtype must be bf16/fp32, got '{value}'")
    if isinstance(value, jnp.dtype):
        return value
    if value in {jnp.bfloat16, jnp.float32}:
        return jnp.dtype(value)
    raise ValueError(f"compute_dtype must be bf16/fp32 or a JAX dtype, got '{value}'")


def _init_linear_normal(
    linear: eqx.nn.Linear,
    key: jax.Array,
    std: float,
) -> eqx.nn.Linear:
    """Initialize a linear layer with normally distributed weights.

    Args:
        linear: Linear layer whose weights are replaced.
        key: PRNG key used to sample weights.
        std: Standard deviation of the normal distribution.

    Returns:
        A copy of ``linear`` with weights sampled from ``N(0, std)``.
    """
    weight = jax.random.normal(key, linear.weight.shape, dtype=linear.weight.dtype) * std
    return eqx.tree_at(lambda layer: layer.weight, linear, weight)


@dataclass(frozen=True)
class LlamaConfig:
    """Configuration for Llama model."""

    vocab_size: int = 256
    dim: int = 384
    depth: int = 6
    heads: int = 6
    dim_head: int = 64
    ffn_hidden_dim: int = 1088
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    share_emb: bool = False
    embed_init_std: float = 0.02  # Embedding initialization std
    init_std: float = 0.02
    compute_dtype: jnp.dtype = jnp.float32  # Dtype for matmul/activation compute


class RMSNorm(eqx.Module):
    """Root Mean Square Layer Normalization.

    Paper: https://arxiv.org/abs/1910.07467
    """

    weight: Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5, *, key: jax.Array):
        """Initialize RMSNorm.

        Args:
            dim: Dimension to normalize.
            eps: Epsilon for numerical stability.
            key: PRNG key (unused, for API consistency).
        """
        del key  # unused
        self.weight = jnp.ones(dim)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        """Apply RMSNorm.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of same shape.
        """
        # Compute in float32 for stability
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.eps)
        normed = x_f32 / rms
        return (normed * self.weight).astype(x.dtype)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> tuple[Array, Array]:
    """Precompute rotary embedding frequencies.

    Args:
        dim: Dimension per head (must be even).
        max_seq_len: Maximum sequence length.
        theta: RoPE theta parameter.

    Returns:
        Tuple of (cos, sin) each of shape [max_seq_len, dim//2].
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))

    # Compute position indices
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)

    # Outer product: [seq_len] x [dim//2] -> [seq_len, dim//2]
    freqs = jnp.outer(positions, inv_freq)

    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rotary_emb(
    x: Array,
    cos: Array,
    sin: Array,
    offset: int = 0,
) -> Array:
    """Apply rotary embeddings to input tensor.

    Args:
        x: Input tensor of shape [B, H, T, D].
        cos: Cosine frequencies of shape [max_seq, D//2].
        sin: Sine frequencies of shape [max_seq, D//2].
        offset: Position offset for caching.

    Returns:
        Tensor with rotary embeddings applied.
    """
    seq_len = x.shape[2]

    # Slice frequencies for current sequence
    cos = cos[offset : offset + seq_len, :].astype(jnp.float32)  # [T, D//2]
    sin = sin[offset : offset + seq_len, :].astype(jnp.float32)  # [T, D//2]

    # Reshape for broadcasting: [T, D//2] -> [1, 1, T, D//2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    # Split x into two halves
    x1, x2 = jnp.split(x, 2, axis=-1)
    x1 = x1.astype(jnp.float32)
    x2 = x2.astype(jnp.float32)

    # Apply rotation
    x_rotated = jnp.concatenate(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        axis=-1,
    )

    return x_rotated.astype(x.dtype)


class CausalSelfAttention(eqx.Module):
    """Multi-head causal self-attention with RoPE."""

    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    wv: eqx.nn.Linear
    wo: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    compute_dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        compute_dtype: jnp.dtype = jnp.float32,
        init_std: float = 0.02,
        *,
        key: jax.Array,
    ):
        """Initialize attention.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            compute_dtype: Dtype for matmul/activation compute.
            init_std: Standard deviation for Gaussian initialization.
            key: PRNG key.
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.compute_dtype = compute_dtype
        inner_dim = num_heads * head_dim

        keys = jax.random.split(key, 4)
        self.wq = eqx.nn.Linear(dim, inner_dim, use_bias=False, key=keys[0])
        self.wk = eqx.nn.Linear(dim, inner_dim, use_bias=False, key=keys[1])
        self.wv = eqx.nn.Linear(dim, inner_dim, use_bias=False, key=keys[2])
        self.wo = eqx.nn.Linear(inner_dim, dim, use_bias=False, key=keys[3])
        self.wq = _init_linear_normal(self.wq, keys[0], init_std)
        self.wk = _init_linear_normal(self.wk, keys[1], init_std)
        self.wv = _init_linear_normal(self.wv, keys[2], init_std)
        self.wo = _init_linear_normal(self.wo, keys[3], init_std)

    def __call__(
        self,
        x: Array,
        cos: Array,
        sin: Array,
        cache: tuple[Array, Array] | None = None,
        return_cache: bool = False,
    ) -> tuple[Array, tuple[Array, Array] | None]:
        """Apply attention.

        Args:
            x: Input tensor of shape [B, T, D].
            cos: Shared cosine RoPE table.
            sin: Shared sine RoPE table.
            cache: Optional KV cache tuple (k_cache, v_cache).
            return_cache: Whether to return updated cache.

        Returns:
            Tuple of (output, cache) where cache is None if not requested.
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = _linear_3d(self.wq, x, self.compute_dtype)  # [B, T, inner_dim]
        k = _linear_3d(self.wk, x, self.compute_dtype)
        v = _linear_3d(self.wv, x, self.compute_dtype)

        # Reshape to [B, H, T, D_head]
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            # During generation, offset by cache length
            offset = cache[0].shape[2]
        else:
            offset = 0

        q = apply_rotary_emb(q, cos, sin, offset)
        k = apply_rotary_emb(k, cos, sin, offset)

        # Update KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = jnp.concatenate([k_cache, k], axis=2)
            v = jnp.concatenate([v_cache, v], axis=2)

        new_cache = (k, v) if return_cache else None

        # Compute attention
        scale = jnp.asarray(self.head_dim**-0.5, dtype=jnp.float32)
        attn_weights = (
            jnp.einsum("bhid,bhjd->bhij", q, k, preferred_element_type=jnp.float32) * scale
        )

        # Causal mask
        q_len, kv_len = q.shape[2], k.shape[2]
        query_positions = jnp.arange(q_len)[:, None] + (kv_len - q_len)
        key_positions = jnp.arange(kv_len)[None, :]
        causal_mask = key_positions <= query_positions
        attn_weights = jnp.where(causal_mask, attn_weights, -jnp.inf)

        attn_probs = jax.nn.softmax(attn_weights, axis=-1).astype(x.dtype)
        out = jnp.einsum(
            "bhij,bhjd->bhid", attn_probs, v, preferred_element_type=jnp.float32
        ).astype(x.dtype)

        # Merge heads and project
        out = out.transpose(0, 2, 1, 3).reshape(batch, -1, self.num_heads * self.head_dim)
        out = _linear_3d(self.wo, out, self.compute_dtype)

        return out, new_cache


class SwiGLU(eqx.Module):
    """SwiGLU feedforward network.

    Uses Swish (SiLU) gated linear unit.
    Paper: https://arxiv.org/abs/2002.05202
    """

    w1: eqx.nn.Linear  # Gate projection
    w2: eqx.nn.Linear  # Down projection
    w3: eqx.nn.Linear  # Up projection
    compute_dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        compute_dtype: jnp.dtype,
        init_std: float = 0.02,
        *,
        key: jax.Array,
    ):
        """Initialize SwiGLU.

        Args:
            dim: Input/output dimension.
            hidden_dim: Hidden dimension.
            compute_dtype: Dtype for matmul/activation compute.
            init_std: Standard deviation for Gaussian initialization.
            key: PRNG key.
        """
        keys = jax.random.split(key, 3)
        self.w1 = eqx.nn.Linear(dim, hidden_dim, use_bias=False, key=keys[0])
        self.w2 = eqx.nn.Linear(hidden_dim, dim, use_bias=False, key=keys[1])
        self.w3 = eqx.nn.Linear(dim, hidden_dim, use_bias=False, key=keys[2])
        self.w1 = _init_linear_normal(self.w1, keys[0], init_std)
        self.w2 = _init_linear_normal(self.w2, keys[1], init_std)
        self.w3 = _init_linear_normal(self.w3, keys[2], init_std)
        self.compute_dtype = compute_dtype

    def __call__(self, x: Array) -> Array:
        """Apply SwiGLU.

        Args:
            x: Input tensor of shape [B, T, D].

        Returns:
            Output tensor of shape [B, T, D].
        """
        gate = _linear_3d(self.w1, x, self.compute_dtype)
        up = _linear_3d(self.w3, x, self.compute_dtype)
        return _linear_3d(self.w2, jax.nn.silu(gate) * up, self.compute_dtype)


class TransformerBlock(eqx.Module):
    """Llama-style transformer block with pre-normalization."""

    attn_norm: RMSNorm
    attn: CausalSelfAttention
    ff_norm: RMSNorm
    ff: SwiGLU

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ffn_hidden_dim: int,
        norm_eps: float = 1e-5,
        compute_dtype: jnp.dtype = jnp.float32,
        init_std: float = 0.02,
        *,
        key: jax.Array,
    ):
        """Initialize transformer block.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            ffn_hidden_dim: FFN hidden dimension.
            norm_eps: RMSNorm epsilon.
            compute_dtype: Dtype for matmul/activation compute.
            init_std: Standard deviation for Gaussian initialization.
            key: PRNG key.
        """
        keys = jax.random.split(key, 4)

        self.attn_norm = RMSNorm(dim, eps=norm_eps, key=keys[0])
        self.attn = CausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            compute_dtype=compute_dtype,
            init_std=init_std,
            key=keys[1],
        )
        self.ff_norm = RMSNorm(dim, eps=norm_eps, key=keys[2])
        self.ff = SwiGLU(
            dim=dim,
            hidden_dim=ffn_hidden_dim,
            compute_dtype=compute_dtype,
            init_std=init_std,
            key=keys[3],
        )

    def __call__(
        self,
        x: Array,
        cos: Array,
        sin: Array,
        cache: tuple[Array, Array] | None = None,
        return_cache: bool = False,
    ) -> tuple[Array, tuple[Array, Array] | None]:
        """Apply transformer block.

        Args:
            x: Input tensor of shape [B, T, D].
            cos: Shared cosine RoPE table.
            sin: Shared sine RoPE table.
            cache: Optional KV cache.
            return_cache: Whether to return cache.

        Returns:
            Tuple of (output, cache).
        """
        # Attention with residual
        normed = jax.vmap(self.attn_norm)(x)
        attn_out, new_cache = self.attn(
            normed,
            cos,
            sin,
            cache=cache,
            return_cache=return_cache,
        )
        x = x + attn_out

        # FFN with residual
        normed = jax.vmap(self.ff_norm)(x)
        x = x + self.ff(normed)

        return x, new_cache


class LlamaLM(eqx.Module):
    """Llama-style causal language model."""

    config: LlamaConfig = eqx.field(static=True)
    embed: eqx.nn.Embedding
    layers: list[TransformerBlock]
    norm: RMSNorm
    lm_head: eqx.nn.Linear | None

    def __init__(self, config: LlamaConfig, *, key: jax.Array):
        """Initialize Llama model.

        Args:
            config: Model configuration.
            key: PRNG key.
        """
        self.config = config
        keys = jax.random.split(key, config.depth + 3)

        # Token embeddings with proper initialization (std=0.02)
        embed = eqx.nn.Embedding(config.vocab_size, config.dim, key=keys[0])
        # Re-initialize with smaller std for stability
        embed_weight = (
            jax.random.normal(keys[0], (config.vocab_size, config.dim)) * config.embed_init_std
        )
        self.embed = eqx.tree_at(lambda e: e.weight, embed, embed_weight)

        # Transformer blocks
        self.layers = [
            TransformerBlock(
                dim=config.dim,
                num_heads=config.heads,
                head_dim=config.dim_head,
                ffn_hidden_dim=config.ffn_hidden_dim,
                norm_eps=config.norm_eps,
                compute_dtype=config.compute_dtype,
                init_std=config.init_std,
                key=keys[i + 1],
            )
            for i in range(config.depth)
        ]

        # Final norm
        self.norm = RMSNorm(config.dim, eps=config.norm_eps, key=keys[-2])

        # Output projection (or None for tied embeddings)
        if config.share_emb:
            self.lm_head = None
        else:
            lm_head = eqx.nn.Linear(config.dim, config.vocab_size, use_bias=False, key=keys[-1])
            self.lm_head = _init_linear_normal(lm_head, keys[-1], config.embed_init_std)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq"],
        cache: list[tuple[Array, Array]] | None = None,
        return_cache: bool = False,
        deterministic: bool = True,
        key: jax.Array | None = None,
    ) -> tuple[Float[Array, "batch seq vocab"], list[tuple[Array, Array]] | None]:
        """Forward pass.

        Args:
            input_ids: Input token IDs of shape [B, T].
            cache: Optional list of KV caches per layer.
            return_cache: Whether to return updated caches.
            deterministic: Whether to use deterministic mode (no dropout).
            key: PRNG key (unused, for API consistency).

        Returns:
            Tuple of (logits, cache) where:
            - logits: Shape [B, T, vocab_size]
            - cache: List of KV caches per layer (or None)
        """
        del deterministic, key  # unused

        # Embed tokens: input_ids [B, T] -> x [B, T, D]
        x = self.embed.weight[input_ids]
        x = x.astype(self.config.compute_dtype)

        cos, sin = precompute_freqs_cis(
            self.config.dim_head,
            self.config.max_seq_len,
            self.config.rope_theta,
        )

        # Apply transformer blocks
        new_caches = [] if return_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, layer_new_cache = layer(
                x,
                cos,
                sin,
                cache=layer_cache,
                return_cache=return_cache,
            )
            if return_cache:
                new_caches.append(layer_new_cache)

        # Final norm
        x = jax.vmap(self.norm)(x)

        # Compute logits
        if self.lm_head is not None:
            logits = jnp.einsum(
                "btd,vd->btv",
                x.astype(jnp.float32),
                self.lm_head.weight.astype(jnp.float32),
                preferred_element_type=jnp.float32,
            )
        else:
            # Tied embeddings: logits = x @ embed.weight.T
            logits = jnp.einsum(
                "btd,vd->btv",
                x.astype(jnp.float32),
                self.embed.weight.astype(jnp.float32),
                preferred_element_type=jnp.float32,
            )

        return logits, new_caches


def build_llama(cfg: dict[str, Any], key: jax.Array) -> LlamaLM:
    """Build Llama model from config dictionary.

    Args:
        cfg: Configuration dictionary.
        key: PRNG key.

    Returns:
        Initialized LlamaLM model.
    """
    config = LlamaConfig(
        vocab_size=cfg.get("num_tokens", 256),
        dim=cfg.get("dim", 384),
        depth=cfg.get("depth", 6),
        heads=cfg.get("heads", 6),
        dim_head=cfg.get("dim_head", 64),
        ffn_hidden_dim=cfg.get("ffn_hidden_dim", 1088),
        max_seq_len=cfg.get("seq_len", 512) * 2,  # Allow generation beyond training length
        rope_theta=cfg.get("rope_theta", 10000.0),
        norm_eps=cfg.get("norm_eps", 1e-5),
        share_emb=cfg.get("share_emb", False),
        init_std=cfg.get("init_std", 0.02),
        compute_dtype=_resolve_compute_dtype(cfg),
    )
    return LlamaLM(config, key=key)


def forward_llama(
    model: LlamaLM,
    input_ids: Int[Array, "batch seq"],
    cache: list[tuple[Array, Array]] | None = None,
    return_cache: bool = False,
    deterministic: bool = True,
    key: jax.Array | None = None,
) -> tuple[Float[Array, "batch seq vocab"], list[tuple[Array, Array]] | None]:
    """Forward pass wrapper for unified interface.

    Args:
        model: LlamaLM model.
        input_ids: Input token IDs.
        cache: Optional KV cache.
        return_cache: Whether to return cache.
        deterministic: Deterministic mode.
        key: PRNG key.

    Returns:
        Tuple of (logits, cache).
    """
    return model(
        input_ids,
        cache=cache,
        return_cache=return_cache,
        deterministic=deterministic,
        key=key,
    )
