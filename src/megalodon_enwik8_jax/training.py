"""Training infrastructure: state, optimizer, and step functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int

from .models import forward_model
from .params import make_trainable_mask

EvalStep = Callable[
    [eqx.Module, Int[Array, "batch seq"], Int[Array, "batch seq"]],
    Float[Array, ""],
]

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
    trainable_mask: Any | None = None,
) -> TrainState:
    """Create initial training state.

    Args:
        model: Equinox model.
        optimizer: Optax optimizer.
        key: Initial PRNG key.
        step: Initial step (default 0).
        trainable_mask: Optional pytree mask of trainable parameters.

    Returns:
        Initialized TrainState.
    """
    if trainable_mask is None:
        trainable_mask = make_trainable_mask(model)

    params, _ = eqx.partition(model, trainable_mask)
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
    trainable_mask: Any,
) -> Callable[
    [TrainState, Int[Array, "accum batch seq"], Int[Array, "accum batch seq"]],
    tuple[TrainState, dict[str, Array]],
]:
    """Create JIT-compiled training step function.

    Uses jax.lax.scan for sequential micro-batches to implement
    true gradient accumulation without scaling memory with A.

    :param dict[str, Any] cfg: Configuration dictionary.
    :param optax.GradientTransformation optimizer: Optax optimizer.
    :param Any trainable_mask: Pytree mask of trainable parameters.
    :return Callable: JIT-compiled train_step function.
    """
    use_jit = cfg.get("jit", True)
    dropout_enabled = any(
        cfg.get(key, 0.0) > 0.0 for key in ("dropout", "attention_dropout", "hidden_dropout")
    )
    deterministic = not dropout_enabled

    def train_step(
        state: TrainState,
        input_ids: Int[Array, "accum batch seq"],
        labels: Int[Array, "accum batch seq"],
    ) -> tuple[TrainState, dict[str, Array]]:
        """Execute one training step with gradient accumulation.

        :param TrainState state: Current training state.
        :param Int[Array, "accum batch seq"] input_ids: Input tokens of shape [A, B, T].
        :param Int[Array, "accum batch seq"] labels: Target labels of shape [A, B, T].
        :return tuple[TrainState, dict[str, Array]]: Updated state and metrics.
        """

        params, static = eqx.partition(state.model, trainable_mask)

        def loss_fn(
            params: eqx.Module,
            batch_input: Int[Array, "batch seq"],
            batch_labels: Int[Array, "batch seq"],
            key: jax.Array,
        ) -> Float[Array, ""]:
            """Compute loss for a single micro-batch."""
            model = eqx.combine(params, static)
            logits, _ = forward_model(model, batch_input, deterministic=deterministic, key=key)
            return cross_entropy_loss(logits, batch_labels)

        def _add_trees(left: Any, right: Any) -> Any:
            if left is None:
                return None
            return left + right

        def _scale_tree(value: Any, scale: float) -> Any:
            if value is None:
                return None
            return value * scale

        def micro_step(
            carry: tuple[jax.Array, Any, jax.Array],
            micro_batch: tuple[Int[Array, "batch seq"], Int[Array, "batch seq"]],
        ) -> tuple[tuple[jax.Array, Any, jax.Array], None]:
            key, grads_accum, loss_accum = carry
            batch_input, batch_labels = micro_batch
            key, subkey = jax.random.split(key)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(
                params, batch_input, batch_labels, subkey
            )
            grads_accum = jax.tree_util.tree_map(
                _add_trees, grads_accum, grads, is_leaf=lambda x: x is None
            )
            loss_accum = loss_accum + loss
            return (key, grads_accum, loss_accum), None

        grads_init = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x) if x is not None else None,
            params,
            is_leaf=lambda x: x is None,
        )
        loss_init = jnp.array(0.0, dtype=jnp.float32)
        (new_key, grads_accum, loss_accum), _ = jax.lax.scan(
            micro_step,
            (state.key, grads_init, loss_init),
            (input_ids, labels),
        )

        scale = 1.0 / input_ids.shape[0]
        loss = loss_accum * scale
        grads = jax.tree_util.tree_map(
            lambda value: _scale_tree(value, scale),
            grads_accum,
            is_leaf=lambda x: x is None,
        )

        # Compute gradient norm before clipping (for logging)
        grad_norm = optax.global_norm(grads)

        # Apply optimizer updates
        updates, new_opt_state = optimizer.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_model = eqx.combine(new_params, static)

        # Update state
        new_state = TrainState(
            step=state.step + 1,
            model=new_model,
            opt_state=new_opt_state,
            key=new_key,
        )

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
        }

        return new_state, metrics

    return eqx.filter_jit(train_step) if use_jit else train_step


# =============================================================================
# Evaluation
# =============================================================================


def make_eval_step(cfg: dict[str, Any]) -> EvalStep:
    """Create JIT-compiled evaluation step function.

    :param dict[str, Any] cfg: Configuration dictionary.
    :return EvalStep: JIT-compiled eval_step function.
    """

    use_jit = cfg.get("jit", True)

    def eval_step(
        model: eqx.Module,
        input_ids: Int[Array, "batch seq"],
        labels: Int[Array, "batch seq"],
    ) -> Float[Array, ""]:
        """Compute validation loss for a batch.

        :param eqx.Module model: Model to evaluate.
        :param Int[Array, "batch seq"] input_ids: Input tokens of shape [B, T].
        :param Int[Array, "batch seq"] labels: Target labels of shape [B, T].
        :return Float[Array, ""]: Scalar loss value.
        """
        logits, _ = forward_model(model, input_ids, deterministic=True)
        return cross_entropy_loss(logits, labels)

    return eqx.filter_jit(eval_step) if use_jit else eval_step


def run_validation(
    model: eqx.Module,
    eval_step: EvalStep,
    val_data: np.ndarray,
    rng: np.random.Generator,
    cfg: dict[str, Any],
) -> Float[Array, ""]:
    """Run validation over multiple batches.

    :param eqx.Module model: Model to evaluate.
    :param EvalStep eval_step: JIT-compiled evaluation step function.
    :param numpy.ndarray val_data: Validation data as uint8 array.
    :param numpy.random.Generator rng: Numpy random generator.
    :param dict[str, Any] cfg: Configuration dictionary.
    :return Float[Array, ""]: Average validation loss.
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
