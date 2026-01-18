"""Text generation with sampling for Llama and Megalodon models."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

from .models import forward_model
from .models.llama import LlamaLM


def apply_temperature(logits: jax.Array, temperature: float) -> jax.Array:
    """Apply temperature scaling to logits."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return logits / temperature


def apply_min_p(logits: jax.Array, min_p: float) -> jax.Array:
    """Apply min-p filtering to logits.

    Masks tokens with probability less than min_p * max_prob.
    """
    if min_p <= 0:
        return logits
    probs = jax.nn.softmax(logits, axis=-1)
    max_probs = probs.max(axis=-1, keepdims=True)
    threshold = min_p * max_probs
    return jnp.where(probs < threshold, -jnp.inf, logits)


def sample_token(
    key: jax.Array,
    logits: jax.Array,
    temperature: float = 1.0,
    min_p: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Sample next token with temperature and min-p filtering.

    Args:
        key: JAX PRNG key.
        logits: Logits for next token, shape [B, V].
        temperature: Sampling temperature.
        min_p: Min-p threshold (0 to disable).

    Returns:
        Tuple of (new_key, token_ids).
    """
    key, sample_key = jax.random.split(key)
    logits = apply_temperature(logits, temperature)
    if min_p > 0:
        logits = apply_min_p(logits, min_p)
    tokens = jax.random.categorical(sample_key, logits, axis=-1)
    return key, tokens


def _generate_llama(
    model: LlamaLM,
    prompt_ids: Int[Array, "batch seq"],
    max_new_tokens: int,
    temperature: float,
    min_p: float,
    key: jax.Array,
) -> Int[Array, "batch total_seq"]:
    """Generate for Llama using Python loop (fast for simple attention)."""
    logits, cache = forward_model(model, prompt_ids, return_cache=True, deterministic=True)

    last_logits = logits[:, -1, :]
    key, next_token = sample_token(key, last_logits, temperature, min_p)
    next_token = next_token[:, None]

    generated = [next_token]

    for _ in range(max_new_tokens - 1):
        logits, cache = forward_model(
            model, next_token, cache=cache, return_cache=True, deterministic=True
        )
        last_logits = logits[:, -1, :]
        key, next_token = sample_token(key, last_logits, temperature, min_p)
        next_token = next_token[:, None]
        generated.append(next_token)

    generated = jnp.concatenate(generated, axis=1)
    return jnp.concatenate([prompt_ids, generated], axis=1)


def generate(
    model: eqx.Module,
    prompt_ids: Int[Array, "batch seq"],
    max_new_tokens: int,
    temperature: float = 1.0,
    min_p: float = 0.0,
    key: jax.Array | None = None,
) -> Int[Array, "batch total_seq"]:
    """Generate text autoregressively with cache-based decoding.

    For Megalodon, uses the megalodon-jax library's generate() which uses
    jax.lax.scan for efficient tracing. For Llama, uses a Python loop
    (still fast due to simpler cache logic).

    Args:
        model: LlamaLM or MegalodonForCausalLM model.
        prompt_ids: Input prompt of shape [B, T].
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        min_p: Min-p filtering threshold (0 to disable).
        key: JAX PRNG key.

    Returns:
        Generated tokens (prompt + new tokens) of shape [B, T + max_new_tokens].
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Dispatch based on model type
    if isinstance(model, LlamaLM):
        return _generate_llama(model, prompt_ids, max_new_tokens, temperature, min_p, key)

    # For Megalodon, use the library's efficient jax.lax.scan-based generate
    from megalodon_jax import generate as megalodon_generate

    # megalodon-jax uses top_p, not min_p. Convert: min_p ~= 1 - top_p
    # For simplicity, if min_p > 0, use top_p = 1 - min_p (approximate)
    top_p = (1.0 - min_p) if min_p > 0 else None

    result, _, _ = megalodon_generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        key=key,
        temperature=temperature,
        top_p=top_p,
    )
    return result
