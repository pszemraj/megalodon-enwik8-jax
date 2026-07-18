"""Smoke tests for training loop."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from megalodon_enwik8_jax.models import build_model
from megalodon_enwik8_jax.utils import (
    build_learning_rate,
    build_optimizer,
    create_train_state,
    load_enwik8,
    make_eval_step,
    make_train_step,
    make_trainable_mask,
    sample_accum_batch,
    sample_batch,
)


def test_optimizer_uses_uniform_decoupled_adamw_decay() -> None:
    """AdamW decays parameters outside the adaptive gradient moments."""
    cfg = {
        "learning_rate": 0.1,
        "weight_decay": 0.2,
        "grad_clip_norm": 0.0,
    }
    optimizer = build_optimizer(cfg)
    params = {"weight": jnp.array([2.0])}
    gradients = {"weight": jnp.array([3.0])}

    updates, _ = optimizer.update(gradients, optimizer.init(params), params)
    updated = optax.apply_updates(params, updates)

    assert jnp.allclose(updated["weight"], jnp.array([1.86]), atol=1e-6)


def test_warmup_cosine_schedule_uses_total_horizon() -> None:
    """Warmup peaks at its boundary and cosine reaches the declared end step."""
    cfg = {
        "learning_rate": 1e-3,
        "lr_schedule": "warmup_cosine",
        "warmup_steps": 5,
        "num_batches": 20,
    }
    schedule = build_learning_rate(cfg)

    assert callable(schedule)
    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(5)) == pytest.approx(1e-3)
    assert float(schedule(20)) == pytest.approx(0.0)


class TestTrainingSmokeTest:
    """Smoke tests for training functionality."""

    def test_llama_loss_decreases(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """Llama training loss decreases over a few steps."""
        # Load data
        train_data, _ = load_enwik8(test_config["data_path"])

        # Build model and optimizer
        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)
        trainable_mask = make_trainable_mask(model)
        optimizer = build_optimizer(test_config)

        # Create training state
        key, state_key = jax.random.split(key)
        state = create_train_state(
            model,
            optimizer,
            state_key,
            step=0,
            trainable_mask=trainable_mask,
        )

        # Create train step
        train_step = make_train_step(test_config, optimizer, trainable_mask)

        # Run a few steps
        losses = []
        for _ in range(5):
            input_ids, labels = sample_accum_batch(
                rng,
                train_data,
                test_config["batch_size"],
                test_config["grad_accum_every"],
                test_config["seq_len"],
            )
            state, metrics = train_step(state, input_ids, labels)
            losses.append(float(metrics["loss"]))

        # Loss should be finite
        assert all(np.isfinite(loss) for loss in losses)

        # Loss should generally decrease (not strictly, but on average)
        # Just check that final loss is less than initial
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"

    def test_llama_grad_norm_finite(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """Gradient norms are finite during Llama training."""
        train_data, _ = load_enwik8(test_config["data_path"])

        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)
        trainable_mask = make_trainable_mask(model)
        optimizer = build_optimizer(test_config)

        key, state_key = jax.random.split(key)
        state = create_train_state(
            model,
            optimizer,
            state_key,
            step=0,
            trainable_mask=trainable_mask,
        )
        train_step = make_train_step(test_config, optimizer, trainable_mask)

        input_ids, labels = sample_accum_batch(
            rng,
            train_data,
            test_config["batch_size"],
            test_config["grad_accum_every"],
            test_config["seq_len"],
        )
        _, metrics = train_step(state, input_ids, labels)

        grad_norm = float(metrics["grad_norm"])
        assert np.isfinite(grad_norm)
        assert grad_norm >= 0

    def test_megalodon_loss_finite(
        self,
        key: jax.Array,
        megalodon_config: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """Megalodon training produces finite loss."""
        train_data, _ = load_enwik8(megalodon_config["data_path"])

        key, model_key = jax.random.split(key)
        model = build_model(megalodon_config, model_key)
        trainable_mask = make_trainable_mask(model)
        optimizer = build_optimizer(megalodon_config)

        key, state_key = jax.random.split(key)
        state = create_train_state(
            model,
            optimizer,
            state_key,
            step=0,
            trainable_mask=trainable_mask,
        )
        train_step = make_train_step(megalodon_config, optimizer, trainable_mask)

        input_ids, labels = sample_accum_batch(
            rng,
            train_data,
            megalodon_config["batch_size"],
            megalodon_config["grad_accum_every"],
            megalodon_config["seq_len"],
        )
        _, metrics = train_step(state, input_ids, labels)

        loss = float(metrics["loss"])
        assert np.isfinite(loss)


class TestEvalStep:
    """Tests for evaluation step."""

    def test_eval_step_returns_scalar(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """eval_step returns scalar loss."""
        train_data, val_data = load_enwik8(test_config["data_path"])

        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)
        eval_step = make_eval_step(test_config)

        input_ids, labels = sample_batch(
            rng, val_data, test_config["batch_size"], test_config["seq_len"]
        )
        loss = eval_step(model, input_ids, labels)

        assert loss.shape == ()  # Scalar
        assert np.isfinite(float(loss))

    def test_eval_deterministic(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """eval_step produces same result for same input."""
        _, val_data = load_enwik8(test_config["data_path"])

        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)
        eval_step = make_eval_step(test_config)

        input_ids, labels = sample_batch(
            rng, val_data, test_config["batch_size"], test_config["seq_len"]
        )

        loss1 = eval_step(model, input_ids, labels)
        loss2 = eval_step(model, input_ids, labels)

        assert float(loss1) == float(loss2)


class TestTrainState:
    """Tests for training state management."""

    def test_state_step_increments(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """Training state step increments after each train_step."""
        train_data, _ = load_enwik8(test_config["data_path"])

        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)
        trainable_mask = make_trainable_mask(model)
        optimizer = build_optimizer(test_config)

        key, state_key = jax.random.split(key)
        state = create_train_state(
            model,
            optimizer,
            state_key,
            step=0,
            trainable_mask=trainable_mask,
        )
        train_step = make_train_step(test_config, optimizer, trainable_mask)

        assert int(state.step) == 0

        input_ids, labels = sample_accum_batch(
            rng,
            train_data,
            test_config["batch_size"],
            test_config["grad_accum_every"],
            test_config["seq_len"],
        )
        state, _ = train_step(state, input_ids, labels)

        assert int(state.step) == 1

    def test_accumulation_uses_exact_target_mean(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
    ) -> None:
        """Duplicating a microbatch leaves the mean loss and update unchanged."""
        model_key, state_key, data_key = jax.random.split(key, 3)
        model = build_model(test_config, model_key)
        trainable_mask = make_trainable_mask(model)
        optimizer = build_optimizer(test_config)
        state = create_train_state(model, optimizer, state_key, trainable_mask=trainable_mask)
        inputs = jax.random.randint(data_key, (1, 2, 8), 0, 256)
        labels = jax.random.randint(data_key, (1, 2, 8), 0, 256)

        one_cfg = {**test_config, "grad_accum_every": 1}
        two_cfg = {**test_config, "grad_accum_every": 2}
        one_state, one_metrics = make_train_step(one_cfg, optimizer, trainable_mask)(
            state, inputs, labels
        )
        two_state, two_metrics = make_train_step(two_cfg, optimizer, trainable_mask)(
            state,
            jnp.concatenate([inputs, inputs]),
            jnp.concatenate([labels, labels]),
        )

        assert jnp.allclose(one_metrics["loss"], two_metrics["loss"], atol=1e-6)
        one_params = jax.tree.leaves(eqx.filter(one_state.model, trainable_mask))
        two_params = jax.tree.leaves(eqx.filter(two_state.model, trainable_mask))
        for one, two in zip(one_params, two_params, strict=True):
            assert jnp.allclose(one, two, atol=2e-6, rtol=2e-6)
