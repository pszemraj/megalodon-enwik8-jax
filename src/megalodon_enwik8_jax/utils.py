"""Utility functions for megalodon-enwik8-jax."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


def count_params(model: eqx.Module) -> int:
    """Count total number of parameters in an Equinox model.

    Args:
        model: Equinox model.

    Returns:
        Total number of parameters (scalar int).
    """
    params, _ = eqx.partition(model, eqx.is_array)
    total = sum(x.size for x in jax.tree.leaves(params))
    return total


def count_trainable_params(model: eqx.Module) -> int:
    """Count trainable parameters (same as count_params for Equinox).

    In Equinox, all array leaves are trainable by default.
    This function is provided for API consistency.

    Args:
        model: Equinox model.

    Returns:
        Number of trainable parameters.
    """
    return count_params(model)


def as_scalar(x: jax.Array | Any) -> float:
    """Convert JAX array to Python scalar.

    Args:
        x: JAX array (scalar) or Python value.

    Returns:
        Python float.
    """
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def ensure_dtype(x: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """Cast array to specified dtype if different.

    Args:
        x: Input array.
        dtype: Target dtype.

    Returns:
        Array cast to dtype (or original if already correct).
    """
    if x.dtype != dtype:
        return x.astype(dtype)
    return x


def format_params(n: int) -> str:
    """Format parameter count with K/M/B suffix.

    Args:
        n: Number of parameters.

    Returns:
        Formatted string like "1.5M" or "340K".
    """
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def model_summary(model: eqx.Module, name: str = "Model") -> str:
    """Generate a simple model summary string.

    Args:
        model: Equinox model.
        name: Display name for the model.

    Returns:
        Summary string with parameter count.
    """
    n_params = count_params(model)
    return f"{name}: {format_params(n_params)} parameters ({n_params:,})"


def tree_map_with_path(
    f,
    tree: Any,
    is_leaf: callable | None = None,
) -> Any:
    """Map function over pytree with access to key path.

    Wrapper around jax.tree_util.tree_map_with_path for convenience.

    Args:
        f: Function taking (key_path, leaf) -> new_leaf.
        tree: Input pytree.
        is_leaf: Optional predicate for what constitutes a leaf.

    Returns:
        Transformed pytree.
    """
    return jax.tree_util.tree_map_with_path(f, tree, is_leaf=is_leaf)
