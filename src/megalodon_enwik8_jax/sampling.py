"""Sampling utilities for text generation."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def apply_temperature(logits: jax.Array, temperature: float) -> jax.Array:
    """Apply temperature scaling to logits.

    Args:
        logits: Unnormalized log probabilities, shape [..., V].
        temperature: Temperature for sampling (higher = more random).

    Returns:
        Scaled logits.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return logits / temperature


def apply_min_p(logits: jax.Array, min_p: float) -> jax.Array:
    """Apply min-p filtering to logits.

    Min-p masks tokens with probability less than min_p * max_prob.
    This is a simple but effective filtering strategy.

    Reference: https://arxiv.org/abs/2407.01082

    Args:
        logits: Unnormalized log probabilities, shape [..., V].
        min_p: Minimum probability threshold relative to max.

    Returns:
        Filtered logits (masked tokens set to -inf).
    """
    if min_p <= 0:
        return logits

    # Compute probabilities
    probs = jax.nn.softmax(logits, axis=-1)
    max_probs = probs.max(axis=-1, keepdims=True)
    threshold = min_p * max_probs

    # Mask tokens below threshold
    return jnp.where(probs < threshold, -jnp.inf, logits)


def apply_top_k(logits: jax.Array, k: int) -> jax.Array:
    """Apply top-k filtering to logits.

    Args:
        logits: Unnormalized log probabilities, shape [..., V].
        k: Number of top tokens to keep.

    Returns:
        Filtered logits (non-top-k tokens set to -inf).
    """
    if k <= 0:
        return logits

    # Get k-th largest value
    top_k_values = jax.lax.top_k(logits, k)[0]
    min_top_k = top_k_values[..., -1:]

    # Mask tokens below threshold
    return jnp.where(logits < min_top_k, -jnp.inf, logits)


def apply_top_p(logits: jax.Array, p: float) -> jax.Array:
    """Apply nucleus (top-p) filtering to logits.

    Args:
        logits: Unnormalized log probabilities, shape [..., V].
        p: Cumulative probability threshold.

    Returns:
        Filtered logits.
    """
    if p >= 1.0:
        return logits

    # Sort in descending order
    sorted_indices = jnp.argsort(-logits, axis=-1)
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)

    # Compute cumulative probabilities
    sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

    # Find cutoff (keep at least one token)
    sorted_mask = cumulative_probs > p
    # Shift mask to keep the first token that exceeds threshold
    sorted_mask = jnp.concatenate(
        [jnp.zeros_like(sorted_mask[..., :1]), sorted_mask[..., :-1]],
        axis=-1,
    )

    # Unsort the mask back to original order
    unsort_indices = jnp.argsort(sorted_indices, axis=-1)
    mask = jnp.take_along_axis(sorted_mask, unsort_indices, axis=-1)

    return jnp.where(mask, -jnp.inf, logits)


def sample_token(key: jax.Array, logits: jax.Array) -> jax.Array:
    """Sample tokens from logits using categorical distribution.

    Args:
        key: JAX PRNG key.
        logits: Unnormalized log probabilities, shape [..., V].

    Returns:
        Sampled token indices, shape [...].
    """
    return jax.random.categorical(key, logits, axis=-1)


def gumbel_sample(
    key: jax.Array,
    logits: jax.Array,
    temperature: float = 1.0,
) -> jax.Array:
    """Sample using Gumbel-max trick.

    Equivalent to categorical sampling but can be more numerically
    stable in some cases.

    Args:
        key: JAX PRNG key.
        logits: Unnormalized log probabilities, shape [..., V].
        temperature: Sampling temperature.

    Returns:
        Sampled token indices, shape [...].
    """
    # Gumbel noise
    uniform = jax.random.uniform(key, logits.shape, minval=1e-10, maxval=1.0)
    gumbel = -jnp.log(-jnp.log(uniform))

    # Perturbed logits
    perturbed = logits / max(temperature, 1e-10) + gumbel

    return jnp.argmax(perturbed, axis=-1)


def sample_next_token(
    key: jax.Array,
    logits: jax.Array,
    temperature: float = 1.0,
    min_p: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Sample next token with configurable filtering.

    Applies filters in order: temperature -> min_p -> top_k -> top_p -> sample.

    Args:
        key: JAX PRNG key.
        logits: Logits for next token, shape [B, V] or [V].
        temperature: Sampling temperature.
        min_p: Min-p threshold (0 to disable).
        top_k: Top-k value (None to disable).
        top_p: Top-p value (None to disable).

    Returns:
        Tuple of (new_key, token_ids).
    """
    key, sample_key = jax.random.split(key)

    # Apply temperature
    logits = apply_temperature(logits, temperature)

    # Apply filters
    if min_p > 0:
        logits = apply_min_p(logits, min_p)
    if top_k is not None and top_k > 0:
        logits = apply_top_k(logits, top_k)
    if top_p is not None and top_p < 1.0:
        logits = apply_top_p(logits, top_p)

    # Sample
    tokens = sample_token(sample_key, logits)

    return key, tokens
