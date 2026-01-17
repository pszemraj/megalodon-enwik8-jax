"""Optimizer construction for training."""

from __future__ import annotations

from typing import Any

import optax


def build_optimizer(cfg: dict[str, Any]) -> optax.GradientTransformation:
    """Build optax optimizer matching PyTorch Adam semantics.

    PyTorch Adam with weight_decay uses coupled L2 regularization
    (not decoupled AdamW). This function replicates that behavior.

    The optimizer chain is:
    1. Gradient clipping (by global norm)
    2. Adam with L2 weight decay
    3. Scale by -lr (optax convention)

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

    # Build optimizer chain
    transforms = []

    # 1. Gradient clipping
    if grad_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip_norm))

    # 2. Adam with L2 weight decay
    # Note: PyTorch's Adam(weight_decay=...) is coupled L2, not AdamW.
    # optax.adamw is decoupled, so we use adam + add_decayed_weights
    # for coupled L2 regularization.
    if weight_decay > 0:
        # Coupled L2: add weight_decay * param to gradient before Adam update
        transforms.append(optax.add_decayed_weights(weight_decay))

    # Adam optimizer
    transforms.append(optax.adam(learning_rate=lr))

    return optax.chain(*transforms)


def build_adamw_optimizer(cfg: dict[str, Any]) -> optax.GradientTransformation:
    """Build optax AdamW optimizer (decoupled weight decay).

    Alternative optimizer with decoupled weight decay, which is
    generally preferred for transformers.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Optax gradient transformation.
    """
    lr = cfg.get("learning_rate", 1e-3)
    weight_decay = cfg.get("weight_decay", 0.0)
    grad_clip_norm = cfg.get("grad_clip_norm", 1.0)

    transforms = []

    if grad_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip_norm))

    transforms.append(optax.adamw(learning_rate=lr, weight_decay=weight_decay))

    return optax.chain(*transforms)
