"""Tests for configuration loading and validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from megalodon_enwik8_jax.utils import validate_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "config_path",
    sorted(CONFIG_DIR.glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_config_has_unique_top_level_keys(config_path: Path) -> None:
    """Tracked configs do not rely on YAML's last-key-wins behavior."""
    document = yaml.compose(config_path.read_text())
    assert isinstance(document, yaml.MappingNode)

    counts = Counter(key.value for key, _ in document.value)
    duplicates = sorted(key for key, count in counts.items() if count > 1)

    assert not duplicates, f"Duplicate keys in {config_path.name}: {duplicates}"


@pytest.mark.parametrize(
    "precision_key",
    [
        "param_dtype",
        "accum_dtype",
        "attention_softmax_dtype",
        "loss_softmax_dtype",
    ],
)
def test_llama_rejects_unsupported_precision(
    precision_key: str,
    test_config: dict[str, Any],
) -> None:
    """Llama configs fail closed for precision controls the model does not expose."""
    test_config[precision_key] = "bf16"

    with pytest.raises(ValueError, match=rf"{precision_key} must be fp32 for Llama baseline"):
        validate_config(test_config)
