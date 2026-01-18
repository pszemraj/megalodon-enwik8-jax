"""GPU smoke tests to verify training and inference use GPU when available."""

from __future__ import annotations

from typing import Any

import jax
import pytest

from megalodon_enwik8_jax.generate import generate
from megalodon_enwik8_jax.models import build_model, forward_model
from megalodon_enwik8_jax.params import make_trainable_mask
from megalodon_enwik8_jax.training import build_optimizer, create_train_state, make_train_step


def _tiny_config() -> dict[str, Any]:
    return {
        "model": "llama",
        "num_tokens": 256,
        "dtype": "bf16",
        "dim": 32,
        "depth": 1,
        "heads": 1,
        "dim_head": 32,
        "ffn_dim_multiplier": 2.0,
        "tied_embedding": True,
        "seq_len": 8,
        "batch_size": 1,
        "grad_accum_every": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "seed": 0,
        "data_path": "data/enwik8.gz",
        "validate_every": 100,
        "val_batches": 2,
        "generate_every": 100,
        "generate_prompt_len": 8,
        "generate_length": 8,
        "save_every": 500,
        "temperature": 1.0,
        "min_p": 0.1,
    }


@pytest.mark.gpu
def test_forward_uses_gpu(gpu_device: jax.Device) -> None:
    """Forward pass places outputs on GPU."""
    with jax.default_device(gpu_device):
        key = jax.random.PRNGKey(0)
        model = build_model(_tiny_config(), key)
        input_ids = jax.random.randint(key, (1, 4), 0, 256)
        logits, _ = forward_model(model, input_ids)

    assert logits.device.platform == "gpu"


@pytest.mark.gpu
def test_generate_uses_gpu(gpu_device: jax.Device) -> None:
    """Generate places outputs on GPU."""
    with jax.default_device(gpu_device):
        key = jax.random.PRNGKey(0)
        model = build_model(_tiny_config(), key)
        prompt_ids = jax.random.randint(key, (1, 4), 0, 256)
        generated = generate(
            model,
            prompt_ids,
            max_new_tokens=2,
            temperature=1.0,
            key=key,
        )

    assert generated.device.platform == "gpu"
    assert generated.shape == (1, 6)


@pytest.mark.gpu
def test_train_step_uses_gpu(gpu_device: jax.Device) -> None:
    """Train step executes on GPU."""
    with jax.default_device(gpu_device):
        key = jax.random.PRNGKey(0)
        cfg = _tiny_config()
        model = build_model(cfg, key)
        trainable_mask = make_trainable_mask(model)
        optimizer = build_optimizer(cfg)

        key, state_key = jax.random.split(key)
        state = create_train_state(
            model,
            optimizer,
            state_key,
            step=0,
            trainable_mask=trainable_mask,
        )
        train_step = make_train_step(cfg, optimizer, trainable_mask)

        grad_accum = cfg["grad_accum_every"]
        batch_size = cfg["batch_size"]
        seq_len = cfg["seq_len"]

        key, data_key = jax.random.split(key)
        input_ids = jax.random.randint(data_key, (grad_accum, batch_size, seq_len), 0, 256)
        labels = jax.random.randint(data_key, (grad_accum, batch_size, seq_len), 0, 256)

        _, metrics = train_step(state, input_ids, labels)

    assert metrics["loss"].device.platform == "gpu"
