"""Tests for model-only run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest
import yaml

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
    assert jnp.array_equal(actual_logits, expected_logits)


def test_legacy_run_metadata_does_not_block_loading(
    test_config: dict[str, Any],
    key: jax.Array,
    tmp_path: Path,
) -> None:
    """Reporting-only fields from completed runs are ignored during inference."""
    cfg = validate_config(test_config)
    model_key, load_key = jax.random.split(key)
    model = build_model(cfg, model_key)
    save_model_artifact(tmp_path, model, cfg)

    config_path = tmp_path / "config.yaml"
    saved_cfg = yaml.safe_load(config_path.read_text())
    saved_cfg.update(
        {
            "comparison_basis": "paper_scaled_width_depth",
            "optimizer": "adamw",
            "min_learning_rate_ratio": 0.0,
        }
    )
    config_path.write_text(yaml.safe_dump(saved_cfg, sort_keys=False))

    _, loaded_cfg = load_model_artifact(tmp_path, load_key)

    assert loaded_cfg == cfg
