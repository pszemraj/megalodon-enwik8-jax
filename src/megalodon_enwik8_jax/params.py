"""Parameter utilities for trainable masks, casting, and dtype checks."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


def make_trainable_mask(model: eqx.Module) -> Any:
    """Create a pytree mask of trainable parameters.

    Excludes non-trainable buffers like Llama RoPE tables.
    """
    mask = jax.tree_util.tree_map(eqx.is_inexact_array, model)

    # Exclude RoPE tables from Llama parameters.
    try:
        from .models.llama import LlamaLM
    except ImportError:
        LlamaLM = None

    if LlamaLM is not None and isinstance(model, LlamaLM):
        layer_count = len(model.layers)
        mask = eqx.tree_at(
            lambda m: [layer.attn.cos for layer in m.layers],
            mask,
            replace=[False] * layer_count,
        )
        mask = eqx.tree_at(
            lambda m: [layer.attn.sin for layer in m.layers],
            mask,
            replace=[False] * layer_count,
        )

    return mask


def cast_trainable(
    model: eqx.Module,
    dtype: jnp.dtype,
    trainable_mask: Any | None = None,
) -> eqx.Module:
    """Cast trainable floating-point parameters to the requested dtype."""
    if trainable_mask is None:
        trainable_mask = make_trainable_mask(model)

    params, static = eqx.partition(model, trainable_mask)

    def _cast_leaf(value: Any) -> Any:
        if value is None:
            return None
        if not jnp.issubdtype(value.dtype, jnp.floating):
            return value
        if value.dtype == dtype:
            return value
        return value.astype(dtype)

    params = jax.tree_util.tree_map(_cast_leaf, params, is_leaf=lambda x: x is None)
    return eqx.combine(params, static)


def assert_trainable_dtype(
    model: eqx.Module,
    dtype: jnp.dtype,
    trainable_mask: Any | None = None,
) -> None:
    """Assert all trainable floating-point parameters match the requested dtype."""
    if trainable_mask is None:
        trainable_mask = make_trainable_mask(model)

    params = eqx.filter(model, trainable_mask)
    leaves = [leaf for leaf in jax.tree.leaves(params) if leaf is not None]

    mismatched = [
        leaf.dtype
        for leaf in leaves
        if jnp.issubdtype(leaf.dtype, jnp.floating) and leaf.dtype != dtype
    ]
    if mismatched:
        unique = sorted({str(item) for item in mismatched})
        raise ValueError(f"Trainable params dtype mismatch. Expected {dtype}, found {unique}.")


def count_trainable_params(
    model: eqx.Module,
    trainable_mask: Any | None = None,
) -> int:
    """Count trainable parameters (excluding non-trainable buffers)."""
    if trainable_mask is None:
        trainable_mask = make_trainable_mask(model)

    params = eqx.filter(model, trainable_mask)
    return sum(leaf.size for leaf in jax.tree.leaves(params) if leaf is not None)


def sample_trainable_dtypes(
    model: eqx.Module,
    trainable_mask: Any | None = None,
    max_samples: int = 3,
) -> list[jnp.dtype]:
    """Return a few representative trainable dtypes for logging."""
    if trainable_mask is None:
        trainable_mask = make_trainable_mask(model)

    params = eqx.filter(model, trainable_mask)
    samples: list[jnp.dtype] = []
    for leaf in jax.tree.leaves(params):
        if leaf is None:
            continue
        samples.append(leaf.dtype)
        if len(samples) >= max_samples:
            break
    return samples
