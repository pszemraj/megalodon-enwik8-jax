"""Training step functions for JAX training loop."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int

from .losses import cross_entropy_loss
from .models import forward_model
from .train_state import TrainState


def make_train_step(
    cfg: dict[str, Any],
    optimizer: optax.GradientTransformation,
):
    """Create JIT-compiled training step function.

    The returned function handles gradient accumulation via jax.lax.scan
    over the accumulation dimension.

    Args:
        cfg: Configuration dictionary.
        optimizer: Optax optimizer.

    Returns:
        JIT-compiled train_step function.
    """

    @eqx.filter_jit
    def train_step(
        state: TrainState,
        input_ids: Int[Array, "accum batch seq"],
        labels: Int[Array, "accum batch seq"],
    ) -> tuple[TrainState, dict[str, Array]]:
        """Execute one training step with gradient accumulation.

        Args:
            state: Current training state.
            input_ids: Input tokens of shape [A, B, T] where A=grad_accum.
            labels: Target labels of shape [A, B, T].

        Returns:
            Tuple of (new_state, metrics) where metrics contains loss, grad_norm.
        """
        grad_accum = input_ids.shape[0]

        def loss_fn(model, batch_input, batch_labels):
            """Compute loss for a single micro-batch."""
            logits, _ = forward_model(model, batch_input, deterministic=True)
            return cross_entropy_loss(logits, batch_labels)

        def accum_step(carry, batch):
            """Accumulate gradients for one micro-batch."""
            grad_sum, loss_sum = carry
            batch_input, batch_labels = batch

            # Compute loss and gradients for this micro-batch
            loss, grads = eqx.filter_value_and_grad(loss_fn)(state.model, batch_input, batch_labels)

            # Accumulate
            new_grad_sum = jax.tree.map(lambda g, gs: g + gs, grads, grad_sum)
            new_loss_sum = loss_sum + loss

            return (new_grad_sum, new_loss_sum), None

        # Initialize gradient accumulator
        params, _ = eqx.partition(state.model, eqx.is_array)
        grad_init = jax.tree.map(jnp.zeros_like, params)
        loss_init = jnp.array(0.0, dtype=jnp.float32)

        # Scan over accumulation steps
        (grad_sum, loss_sum), _ = jax.lax.scan(
            accum_step,
            (grad_init, loss_init),
            (input_ids, labels),
        )

        # Normalize gradients by accumulation steps
        grads = jax.tree.map(lambda g: g / grad_accum, grad_sum)
        avg_loss = loss_sum / grad_accum

        # Compute gradient norm before clipping (for logging)
        grad_norm = optax.global_norm(grads)

        # Apply optimizer updates
        params, static = eqx.partition(state.model, eqx.is_array)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_model = eqx.combine(new_params, static)

        # Update state
        new_state = TrainState(
            step=state.step + 1,
            model=new_model,
            opt_state=new_opt_state,
            key=state.key,
        )

        metrics = {
            "loss": avg_loss,
            "grad_norm": grad_norm,
        }

        return new_state, metrics

    return train_step


def make_eval_step(cfg: dict[str, Any]):
    """Create JIT-compiled evaluation step function.

    Args:
        cfg: Configuration dictionary.

    Returns:
        JIT-compiled eval_step function.
    """

    @eqx.filter_jit
    def eval_step(
        model: eqx.Module,
        input_ids: Int[Array, "batch seq"],
        labels: Int[Array, "batch seq"],
    ) -> Float[Array, ""]:
        """Compute validation loss for a batch.

        Args:
            model: Model to evaluate.
            input_ids: Input tokens of shape [B, T].
            labels: Target labels of shape [B, T].

        Returns:
            Scalar loss value.
        """
        logits, _ = forward_model(model, input_ids, deterministic=True)
        return cross_entropy_loss(logits, labels)

    return eval_step


def run_validation(
    model: eqx.Module,
    eval_step,
    val_data,
    rng: Any,
    cfg: dict[str, Any],
) -> Float[Array, ""]:
    """Run validation over multiple batches.

    Args:
        model: Model to evaluate.
        eval_step: JIT-compiled evaluation step.
        val_data: Validation data (numpy uint8 array).
        rng: Numpy random generator.
        cfg: Configuration dictionary.

    Returns:
        Average validation loss.
    """
    from .data import sample_batch

    val_batches = cfg.get("val_batches", 10)
    batch_size = cfg.get("batch_size", 1)
    seq_len = cfg.get("seq_len", 512)

    total_loss = jnp.array(0.0, dtype=jnp.float32)

    for _ in range(val_batches):
        input_ids, labels = sample_batch(rng, val_data, batch_size, seq_len)
        loss = eval_step(model, input_ids, labels)
        total_loss = total_loss + loss

    return total_loss / val_batches
