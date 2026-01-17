"""Loss functions for language modeling."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute cross-entropy loss for language modeling.

    Args:
        logits: Predicted logits of shape [B, T, V] (any dtype).
        labels: Ground truth labels of shape [B, T] (int32).

    Returns:
        Scalar loss value in float32, averaged over all tokens.
    """
    # Compute in float32 for numerical stability
    logits_f32 = logits.astype(jnp.float32)

    # Log-softmax for numerical stability
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)

    # Gather log probabilities for target tokens
    # Shape: [B, T]
    batch_size, seq_len, vocab_size = logits.shape
    batch_idx = jnp.arange(batch_size)[:, None]
    seq_idx = jnp.arange(seq_len)[None, :]
    target_log_probs = log_probs[batch_idx, seq_idx, labels]

    # Average negative log probability
    loss = -target_log_probs.mean()

    return loss


def cross_entropy_loss_unreduced(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute unreduced cross-entropy loss (per-token).

    Useful for gradient accumulation where we sum losses across
    micro-batches before normalizing.

    Args:
        logits: Predicted logits of shape [B, T, V].
        labels: Ground truth labels of shape [B, T].

    Returns:
        Per-token losses of shape [B, T] in float32.
    """
    logits_f32 = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)

    batch_size, seq_len, _ = logits.shape
    batch_idx = jnp.arange(batch_size)[:, None]
    seq_idx = jnp.arange(seq_len)[None, :]
    target_log_probs = log_probs[batch_idx, seq_idx, labels]

    return -target_log_probs


def bpc_from_loss(loss: jax.Array) -> jax.Array:
    """Convert cross-entropy loss to bits-per-character.

    BPC is a standard metric for character-level language models.
    Lower is better; random guessing on enwik8 (256 tokens) = 8.0 BPC.

    Args:
        loss: Cross-entropy loss (nats).

    Returns:
        Bits-per-character value.
    """
    # loss / ln(2) converts nats to bits
    return loss / jnp.log(2.0)
