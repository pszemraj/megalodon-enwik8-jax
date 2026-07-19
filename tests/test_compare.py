"""Tests for the paired experiment runner."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest
import yaml

import compare
from compare import _aggregate

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_main_runs_or_aggregates_only_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner alternates launches and can rebuild only the aggregate summary."""
    calls: list[tuple[str, int]] = []

    def fake_run(config: Path, run_dir: Path, seed: int) -> None:
        calls.append((run_dir.name, seed))

    monkeypatch.setattr(compare, "_run_one", fake_run)
    monkeypatch.setattr(compare, "_aggregate", lambda output_dir, seeds: {"seeds": list(seeds)})
    output_dir = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare.py",
            "--output-dir",
            str(output_dir),
            "--seeds",
            "7",
            "17",
        ],
    )

    compare.main()

    assert calls == [
        ("megalodon", 7),
        ("llama", 7),
        ("llama", 17),
        ("megalodon", 17),
    ]
    assert json.loads((output_dir / "comparison.json").read_text()) == {"seeds": [7, 17]}

    calls.clear()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--aggregate-only"])

    compare.main()

    assert calls == []
    assert json.loads((output_dir / "comparison.json").read_text()) == {"seeds": [7, 17]}


@pytest.mark.parametrize("invalid_pair", ["swapped", "batch_size"], ids=str)
def test_main_rejects_invalid_pair_before_launch(
    invalid_pair: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model identity and shared-protocol errors fail before creating run output."""
    megalodon_path = CONFIG_DIR / "megalodon_paper_scaled_512.yaml"
    llama_path = CONFIG_DIR / "llama2_paper_scaled_512.yaml"
    expected_error = "expected 'megalodon'"
    if invalid_pair == "swapped":
        megalodon_path = llama_path
    else:
        llama_config = yaml.safe_load(llama_path.read_text())
        llama_config["batch_size"] = 64
        llama_path = tmp_path / "mismatched_llama.yaml"
        llama_path.write_text(yaml.safe_dump(llama_config))
        expected_error = "batch_size"

    output_dir = tmp_path / "runs"
    monkeypatch.setattr(
        compare,
        "_run_one",
        lambda *args: pytest.fail("training launched before config validation"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare.py",
            "--megalodon-config",
            str(megalodon_path),
            "--llama-config",
            str(llama_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(ValueError, match=expected_error):
        compare.main()

    assert not output_dir.exists()


def test_aggregate_reports_per_seed_and_paired_statistics(tmp_path: Path) -> None:
    """Aggregation uses only the final summaries produced by training."""
    seeds = (7, 17, 42)
    for index, seed in enumerate(seeds):
        for model in ("megalodon", "llama"):
            run_dir = tmp_path / f"seed_{seed}" / model
            run_dir.mkdir(parents=True)
            model_offset = 0.0 if model == "megalodon" else 0.2
            summary = {
                "parameter_count": 1_000_000 if model == "megalodon" else 995_000,
                "final_validation_loss": 1.0 + index * 0.1 + model_offset,
                "final_validation_bpc": 1.5 + index * 0.1 + model_offset,
                "steady_state": {
                    "step_seconds_median": 0.1 + model_offset / 10,
                    (
                        "tokens_per_second_aggregate" if seed == seeds[0] else "tokens_per_second"
                    ): 1000.0 - model_offset * 100,
                },
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))

    result = _aggregate(tmp_path, seeds)

    loss_deltas = [-0.2, -0.2, -0.2]
    assert result["parameter_counts"] == {"megalodon": 1_000_000, "llama": 995_000}
    assert result["paired_delta_summary"]["validation_loss"]["mean"] == pytest.approx(
        statistics.mean(loss_deltas)
    )
    assert result["paired_delta_summary"]["validation_loss"]["sample_std"] == pytest.approx(
        statistics.stdev(loss_deltas)
    )
    assert [pair["seed"] for pair in result["paired_results"]] == list(seeds)
    assert result["paired_results"][0]["megalodon"]["validation_loss"] == pytest.approx(1.0)
