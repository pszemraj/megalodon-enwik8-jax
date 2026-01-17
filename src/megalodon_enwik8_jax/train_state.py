"""Training state container for JAX training loops."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class TrainState(eqx.Module):
    """Container for all mutable training state.

    Using eqx.Module makes this a valid PyTree for JIT compilation.

    Attributes:
        step: Current training step (scalar int32).
        model: Equinox model (pytree of arrays).
        opt_state: Optax optimizer state.
        key: JAX PRNG key for sampling/dropout.
    """

    step: jax.Array  # scalar int32
    model: eqx.Module
    opt_state: optax.OptState
    key: jax.Array  # PRNGKey


def create_train_state(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    key: jax.Array,
    step: int = 0,
) -> TrainState:
    """Create initial training state.

    Args:
        model: Equinox model.
        optimizer: Optax optimizer.
        key: Initial PRNG key.
        step: Initial step (default 0).

    Returns:
        Initialized TrainState.
    """
    # Initialize optimizer state
    params, _ = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)

    return TrainState(
        step=jnp.array(step, dtype=jnp.int32),
        model=model,
        opt_state=opt_state,
        key=key,
    )


def get_params(state: TrainState) -> Any:
    """Extract trainable parameters from state.

    Args:
        state: Training state.

    Returns:
        Pytree of model parameters.
    """
    params, _ = eqx.partition(state.model, eqx.is_array)
    return params


def update_train_state(
    state: TrainState,
    *,
    step: jax.Array | None = None,
    model: eqx.Module | None = None,
    opt_state: optax.OptState | None = None,
    key: jax.Array | None = None,
) -> TrainState:
    """Create a new TrainState with updated fields.

    Args:
        state: Current state.
        step: New step (or None to keep current).
        model: New model (or None to keep current).
        opt_state: New optimizer state (or None to keep current).
        key: New key (or None to keep current).

    Returns:
        New TrainState with updated fields.
    """
    return TrainState(
        step=step if step is not None else state.step,
        model=model if model is not None else state.model,
        opt_state=opt_state if opt_state is not None else state.opt_state,
        key=key if key is not None else state.key,
    )
