"""Tests for configuration loading and validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

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
