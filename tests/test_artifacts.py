"""Tests for model-only run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from megalodon_enwik8_jax.models import build_model, forward_model
from megalodon_enwik8_jax.utils import (
    load_model_artifact,
    save_model_artifact,
    validate_config,
)


@pytest.mark.parametrize("fixture_name", ["test_config", "megalodon_config"])
def test_model_artifact_roundtrip(
    request: pytest.FixtureRequest,
    fixture_name: str,
    key: jax.Array,
    tmp_path: Path,
) -> None:
    """Both payload formats reproduce model logits and resolved config."""
    cfg: dict[str, Any] = validate_config(request.getfixturevalue(fixture_name))
    key, model_key, load_key, data_key = jax.random.split(key, 4)
    model = build_model(cfg, model_key)
    input_ids = jax.random.randint(data_key, (1, 8), 0, 256)
    expected_logits, _ = forward_model(model, input_ids)

    payload = save_model_artifact(tmp_path, model, cfg)
    loaded_model, loaded_cfg = load_model_artifact(tmp_path, load_key)
    actual_logits, _ = forward_model(loaded_model, input_ids)

    assert payload.name == ("model.eqx" if cfg["model"] == "llama" else "model.safetensors")
    assert loaded_cfg == cfg
    assert jnp.allclose(actual_logits, expected_logits, atol=2e-2, rtol=2e-2)
