"""Text generation for both Llama and Megalodon models."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

from .models import forward_model, get_model_type
from .models.llama import LlamaLM
from .sampling import sample_next_token


def generate(
    model: eqx.Module,
    prompt_ids: Int[Array, "batch seq"],
    max_new_tokens: int,
    temperature: float = 1.0,
    min_p: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    key: jax.Array = None,
) -> Int[Array, "batch total_seq"]:
    """Generate text autoregressively.

    Uses cache-based decoding for efficient generation:
    1. Prefill: Process prompt to get initial cache
    2. Decode: Generate tokens one at a time using cached KV

    Args:
        model: LlamaLM or MegalodonForCausalLM model.
        prompt_ids: Input prompt of shape [B, T].
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        min_p: Min-p filtering threshold (0 to disable).
        top_k: Top-k filtering (None to disable).
        top_p: Top-p (nucleus) filtering (None to disable).
        key: JAX PRNG key.

    Returns:
        Generated tokens (prompt + new tokens) of shape [B, T + max_new_tokens].
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    model_type = get_model_type(model)

    if model_type == "llama":
        return _generate_llama(
            model, prompt_ids, max_new_tokens, temperature, min_p, top_k, top_p, key
        )
    else:
        return _generate_megalodon(
            model, prompt_ids, max_new_tokens, temperature, min_p, top_k, top_p, key
        )


def _generate_llama(
    model: LlamaLM,
    prompt_ids: Int[Array, "batch seq"],
    max_new_tokens: int,
    temperature: float,
    min_p: float,
    top_k: int | None,
    top_p: float | None,
    key: jax.Array,
) -> Int[Array, "batch total_seq"]:
    """Generate using Llama with KV cache.

    Args:
        model: LlamaLM model.
        prompt_ids: Input prompt.
        max_new_tokens: Tokens to generate.
        temperature: Sampling temperature.
        min_p: Min-p threshold.
        top_k: Top-k value.
        top_p: Top-p value.
        key: PRNG key.

    Returns:
        Generated sequence.
    """
    # Prefill: process prompt
    logits, cache = forward_model(model, prompt_ids, return_cache=True, deterministic=True)

    # Get last token logits and sample first new token
    last_logits = logits[:, -1, :]  # [B, V]
    key, next_token = sample_next_token(key, last_logits, temperature, min_p, top_k, top_p)
    next_token = next_token[:, None]  # [B, 1]

    # Collect generated tokens
    generated = [next_token]

    # Decode loop
    for _ in range(max_new_tokens - 1):
        # Forward with cache
        logits, cache = forward_model(
            model, next_token, cache=cache, return_cache=True, deterministic=True
        )

        # Sample next token
        last_logits = logits[:, -1, :]
        key, next_token = sample_next_token(key, last_logits, temperature, min_p, top_k, top_p)
        next_token = next_token[:, None]
        generated.append(next_token)

    # Concatenate all generated tokens
    generated = jnp.concatenate(generated, axis=1)

    # Return prompt + generated
    return jnp.concatenate([prompt_ids, generated], axis=1)


def _generate_megalodon(
    model: Any,
    prompt_ids: Int[Array, "batch seq"],
    max_new_tokens: int,
    temperature: float,
    min_p: float,
    top_k: int | None,
    top_p: float | None,
    key: jax.Array,
) -> Int[Array, "batch total_seq"]:
    """Generate using Megalodon with cache.

    CRITICAL: Megalodon MUST use cache-based decode to properly
    exercise CEMA streaming and TimestepNorm state updates.

    Args:
        model: MegalodonForCausalLM model.
        prompt_ids: Input prompt.
        max_new_tokens: Tokens to generate.
        temperature: Sampling temperature.
        min_p: Min-p threshold.
        top_k: Top-k value.
        top_p: Top-p value.
        key: PRNG key.

    Returns:
        Generated sequence.
    """
    # Prefill: process prompt
    logits, cache = forward_model(model, prompt_ids, return_cache=True, deterministic=True)

    # Get last token logits and sample first new token
    last_logits = logits[:, -1, :]  # [B, V]
    key, next_token = sample_next_token(key, last_logits, temperature, min_p, top_k, top_p)
    next_token = next_token[:, None]  # [B, 1]

    # Collect generated tokens
    generated = [next_token]

    # Decode loop
    for _ in range(max_new_tokens - 1):
        # Forward with cache
        logits, cache = forward_model(
            model, next_token, cache=cache, return_cache=True, deterministic=True
        )

        # Sample next token
        last_logits = logits[:, -1, :]
        key, next_token = sample_next_token(key, last_logits, temperature, min_p, top_k, top_p)
        next_token = next_token[:, None]
        generated.append(next_token)

    # Concatenate all generated tokens
    generated = jnp.concatenate(generated, axis=1)

    # Return prompt + generated
    return jnp.concatenate([prompt_ids, generated], axis=1)


def generate_greedy(
    model: eqx.Module,
    prompt_ids: Int[Array, "batch seq"],
    max_new_tokens: int,
) -> Int[Array, "batch total_seq"]:
    """Generate text using greedy decoding (no sampling).

    Args:
        model: Model instance.
        prompt_ids: Input prompt.
        max_new_tokens: Tokens to generate.

    Returns:
        Generated sequence.
    """
    # Prefill
    logits, cache = forward_model(model, prompt_ids, return_cache=True, deterministic=True)

    # Greedy decode
    next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    generated = [next_token]

    for _ in range(max_new_tokens - 1):
        logits, cache = forward_model(
            model, next_token, cache=cache, return_cache=True, deterministic=True
        )
        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        generated.append(next_token)

    generated = jnp.concatenate(generated, axis=1)
    return jnp.concatenate([prompt_ids, generated], axis=1)
