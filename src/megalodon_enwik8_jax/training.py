"""Training infrastructure: state, optimizer, and step functions."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int

from .models import forward_model

# =============================================================================
# TrainState
# =============================================================================


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
    params, _ = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)

    return TrainState(
        step=jnp.array(step, dtype=jnp.int32),
        model=model,
        opt_state=opt_state,
        key=key,
    )


# =============================================================================
# Optimizer
# =============================================================================


def build_optimizer(cfg: dict[str, Any]) -> optax.GradientTransformation:
    """Build optax optimizer matching PyTorch Adam semantics.

    PyTorch Adam with weight_decay uses coupled L2 regularization
    (not decoupled AdamW). This function replicates that behavior.

    Args:
        cfg: Configuration dictionary containing:
            - learning_rate: Learning rate (default 1e-3).
            - weight_decay: L2 weight decay (default 0.0).
            - grad_clip_norm: Max gradient norm (default 1.0).

    Returns:
        Optax gradient transformation.
    """
    lr = cfg.get("learning_rate", 1e-3)
    weight_decay = cfg.get("weight_decay", 0.0)
    grad_clip_norm = cfg.get("grad_clip_norm", 1.0)

    transforms = []

    # Gradient clipping
    if grad_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip_norm))

    # Coupled L2 weight decay (PyTorch Adam behavior)
    if weight_decay > 0:
        transforms.append(optax.add_decayed_weights(weight_decay))

    # Adam optimizer
    transforms.append(optax.adam(learning_rate=lr))

    return optax.chain(*transforms)


# =============================================================================
# Loss functions
# =============================================================================


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute cross-entropy loss for language modeling.

    Args:
        logits: Predicted logits of shape [B, T, V] (any dtype).
        labels: Ground truth labels of shape [B, T] (int32).

    Returns:
        Scalar loss value in float32, averaged over all tokens.
    """
    logits_f32 = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)

    batch_size, seq_len, vocab_size = logits.shape
    batch_idx = jnp.arange(batch_size)[:, None]
    seq_idx = jnp.arange(seq_len)[None, :]
    target_log_probs = log_probs[batch_idx, seq_idx, labels]

    return -target_log_probs.mean()


def bpc_from_loss(loss: jax.Array) -> jax.Array:
    """Convert cross-entropy loss to bits-per-character."""
    return loss / jnp.log(2.0)


# =============================================================================
# Training step
# =============================================================================


def make_train_step(
    cfg: dict[str, Any],
    optimizer: optax.GradientTransformation,
):
    """Create JIT-compiled training step function.

    Uses vmap for parallel loss computation across micro-batches,
    then single gradient computation on the averaged loss.

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

        def loss_fn(model, all_inputs, all_labels):
            """Compute average loss across all micro-batches."""

            def single_batch_loss(batch_input, batch_labels):
                logits, _ = forward_model(model, batch_input, deterministic=True)
                return cross_entropy_loss(logits, batch_labels)

            # vmap over the accumulation dimension
            losses = jax.vmap(single_batch_loss)(all_inputs, all_labels)
            return losses.mean()

        # Single gradient computation on averaged loss
        loss, grads = eqx.filter_value_and_grad(loss_fn)(state.model, input_ids, labels)

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
            "loss": loss,
            "grad_norm": grad_norm,
        }

        return new_state, metrics

    return train_step


# =============================================================================
# Evaluation
# =============================================================================


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
