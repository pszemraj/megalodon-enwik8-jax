"""Tests for paired experiment validation and aggregation."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

import compare
from compare import (
    _aggregate,
    _interpolate_validation_at_flops,
    _is_complete,
    _run_one,
    _validate_protocol,
)
from megalodon_enwik8_jax.utils import validate_config, write_yaml_atomic


def _assert_mean_std(actual: dict[str, float], values: list[float]) -> None:
    """Check a serialized aggregate against an independent stdlib calculation."""
    assert actual["mean"] == pytest.approx(statistics.mean(values))
    assert actual["sample_std"] == pytest.approx(statistics.stdev(values))


def test_protocol_mismatch_fails_before_training() -> None:
    mega = {"model": "megalodon", "share_emb": False, "seq_len": 512}
    llama = {"model": "llama", "share_emb": False, "seq_len": 256}

    with pytest.raises(ValueError, match="protocol mismatch"):
        _validate_protocol(mega, llama)


def test_paper_scaled_protocol_allows_model_specific_learning_rates() -> None:
    mega = {
        "model": "megalodon",
        "comparison_basis": "paper_scaled_width_depth",
        "share_emb": False,
        "model_dim": 384,
        "num_layers": 6,
        "num_heads": 1,
        "z_dim": 96,
        "value_dim": 768,
        "ffn_hidden_dim": 768,
        "cema_ndim": 16,
        "chunk_size": 512,
        "norm_num_groups": 12,
        "rope_base": 100_000.0,
        "swiglu": True,
        "seq_len": 512,
        "learning_rate": 3.5e-4,
    }
    llama = {
        "model": "llama",
        "comparison_basis": "paper_scaled_width_depth",
        "share_emb": False,
        "dim": 384,
        "depth": 6,
        "heads": 3,
        "dim_head": 128,
        "ffn_hidden_dim": 1024,
        "rope_theta": 10_000.0,
        "seq_len": 512,
        "learning_rate": 5e-4,
    }

    _validate_protocol(mega, llama)


def test_main_records_validated_protocol_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded protocol matches defaults applied by train.py."""
    config_paths: dict[str, Path] = {}
    sources = {
        "megalodon": Path("configs/megalodon_paper_scaled_512.yaml"),
        "llama": Path("configs/llama2_paper_scaled_512.yaml"),
    }
    for model, source in sources.items():
        config = yaml.safe_load(source.read_text())
        config.pop("num_batches")
        path = tmp_path / source.name
        path.write_text(yaml.safe_dump(config))
        config_paths[model] = path

    captured: dict[str, Any] = {}

    def fake_aggregate(
        output_dir: Path, seeds: tuple[int, ...], protocol: dict[str, Any]
    ) -> dict[str, Any]:
        captured.update(output_dir=output_dir, seeds=seeds, protocol=protocol)
        return {}

    def fake_write(path: Path, value: dict[str, Any]) -> None:
        captured.update(results_path=path, result=value)

    output_dir = tmp_path / "runs"
    monkeypatch.setattr(compare, "_run_one", lambda *_: None)
    monkeypatch.setattr(compare, "_aggregate", fake_aggregate)
    monkeypatch.setattr(compare, "_write_json_atomic", fake_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare.py",
            "--megalodon-config",
            str(config_paths["megalodon"]),
            "--llama-config",
            str(config_paths["llama"]),
            "--output-dir",
            str(output_dir),
            "--seeds",
            "7",
        ],
    )

    compare.main()

    assert captured["protocol"]["num_batches"] == 1200
    assert captured["results_path"] == output_dir / "comparison.json"


def test_compute_matched_validation_interpolates_between_checkpoints() -> None:
    validation = [
        {
            "step": 0,
            "tokens_seen": 0,
            "loss": 2.0,
            "bpc": 3.0,
            "xla_estimated_training_flops": 0.0,
        },
        {
            "step": 100,
            "tokens_seen": 1000,
            "loss": 1.0,
            "bpc": 1.5,
            "xla_estimated_training_flops": 200.0,
        },
    ]

    endpoint = _interpolate_validation_at_flops(validation, 50.0)

    assert endpoint["interpolated_step"] == pytest.approx(25.0)
    assert endpoint["interpolated_tokens_seen"] == pytest.approx(250.0)
    assert endpoint["loss"] == pytest.approx(1.75)
    assert endpoint["bpc"] == pytest.approx(2.625)

    shifted_validation = [
        {**item, "xla_estimated_training_flops": item["xla_estimated_training_flops"] + 10.0}
        for item in validation
    ]
    with pytest.raises(ValueError, match=r"trajectory \[10\.0, 210\.0\]"):
        _interpolate_validation_at_flops(shifted_validation, 5.0)


def test_completed_run_reuses_normalized_config_across_path_spellings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    relative_run_dir = Path("runs/run")
    run_dir = relative_run_dir.resolve()
    run_dir.mkdir(parents=True)
    raw_config = {
        "model": "llama",
        "num_batches": 10,
        "validate_every": 5,
    }
    input_config_path = tmp_path / "input.yaml"
    with open(input_config_path, "w") as file:
        yaml.safe_dump(raw_config, file)
    config = validate_config(raw_config.copy())
    config["seed"] = 7
    config["run_dir"] = str(relative_run_dir)
    config_path = run_dir / "config.yaml"
    model_path = run_dir / "model.eqx"
    write_yaml_atomic(config_path, config)
    model_path.write_bytes(b"model")
    with open(run_dir / "summary.json", "w") as file:
        json.dump({"model": "llama", "seed": 7, "completed_steps": 10}, file)
    with open(run_dir / "metrics.jsonl", "w") as file:
        file.write(json.dumps({"kind": "validation", "step": 0}) + "\n")
        for step in range(1, 11):
            file.write(json.dumps({"kind": "train", "step": step}) + "\n")
            if step % 5 == 0:
                file.write(json.dumps({"kind": "validation", "step": step}) + "\n")
    with open(run_dir / "manifest.json", "w") as file:
        json.dump(
            {
                "model_type": "llama",
                "model_file": "model.eqx",
                "seed": 7,
                "training_step": 10,
            },
            file,
        )

    expected_config = {**raw_config, "seed": 7, "run_dir": str(run_dir)}
    assert _is_complete(run_dir, expected_config)
    assert not _is_complete(run_dir, {**expected_config, "seed": 17})
    _run_one(input_config_path, run_dir, 7)


def test_aggregate_reports_paired_mean_and_sample_std(tmp_path: Path) -> None:
    seeds = (7, 17, 42)
    protocol = {"num_batches": 10}
    for index, seed in enumerate(seeds):
        for model in ("megalodon", "llama"):
            run_dir = tmp_path / f"seed_{seed}" / model
            run_dir.mkdir(parents=True)
            model_offset = 0.0 if model == "megalodon" else 0.2
            summary = {
                "seed": seed,
                "model": model,
                "completed_steps": 10,
                "parameter_count": 1_000_000 if model == "megalodon" else 995_000,
                "final_validation_loss": 1.0 + index * 0.1 + model_offset,
                "final_validation_bpc": 1.5 + index * 0.1 + model_offset,
                "steady_state": {
                    "step_seconds_median": 0.1 + model_offset / 10,
                    "step_seconds_p90": 0.12 + model_offset / 10,
                    "tokens_per_second_aggregate": 1000.0 - model_offset * 100,
                },
                "cold_compile_seconds": {"train_step": 2.0, "eval_step": 1.0},
                "xla_estimated_flops_per_step": None,
                "xla_estimated_training_flops": None,
            }
            with open(run_dir / "summary.json", "w") as file:
                json.dump(summary, file)
            with open(run_dir / "manifest.json", "w") as file:
                json.dump({"model_file": "model.eqx"}, file)
            with open(run_dir / "metrics.jsonl", "w") as file:
                for step in (0, 10):
                    file.write(
                        json.dumps(
                            {
                                "kind": "validation",
                                "step": step,
                                "tokens_seen": step * 100,
                                "loss": 1.0 + model_offset,
                                "bpc": 1.5 + model_offset,
                            }
                        )
                        + "\n"
                    )

    result = _aggregate(tmp_path, seeds, protocol)

    assert result["architecture_comparison"]["relative_gap"] < 0.01
    assert result["paired_delta_summary"]["validation_loss"]["mean"] == pytest.approx(-0.2)
    assert result["paired_delta_summary"]["validation_loss"]["sample_std"] == pytest.approx(0.0)
    assert result["compute_matched_secondary"] is None
    for model in ("megalodon", "llama"):
        assert result["model_metrics"][model]["xla_estimated_flops_per_step"] is None
        assert result["model_metrics"][model]["xla_estimated_training_flops"] is None
    assert [item["seed"] for item in result["execution_order"]] == list(seeds)
    assert result["runs"]["megalodon_seed_7"]["validation"][-1]["step"] == 10
