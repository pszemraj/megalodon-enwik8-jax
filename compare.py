#!/usr/bin/env python3
"""Run and aggregate the predeclared paired Megalodon/Llama comparison."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from megalodon_enwik8_jax.utils import validate_config

DEFAULT_SEEDS = (7, 17, 42)
COMMON_PROTOCOL_KEYS = (
    "comparison_basis",
    "num_tokens",
    "share_emb",
    "param_dtype",
    "compute_dtype",
    "accum_dtype",
    "attention_softmax_dtype",
    "loss_softmax_dtype",
    "jit",
    "num_batches",
    "batch_size",
    "grad_accum_every",
    "optimizer",
    "lr_schedule",
    "warmup_steps",
    "min_learning_rate_ratio",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "weight_decay",
    "grad_clip_norm",
    "data_path",
    "seq_len",
    "validate_every",
    "val_batch_size",
    "val_batches",
)


def _validate_paper_scaled_geometry(
    megalodon_cfg: dict[str, Any], llama_cfg: dict[str, Any]
) -> None:
    """Validate the declared 7B-paper ratios at the reduced model width.

    Args:
        megalodon_cfg: Megalodon experiment configuration.
        llama_cfg: Llama experiment configuration.

    Raises:
        ValueError: If the configurations do not follow the paper-scaled geometry.
    """
    model_dim = megalodon_cfg.get("model_dim")
    llama_dim = llama_cfg.get("dim")
    errors: list[str] = []
    if model_dim != llama_dim:
        errors.append(f"model width differs ({model_dim} versus {llama_dim})")
    if megalodon_cfg.get("num_layers") != llama_cfg.get("depth"):
        errors.append("model depth differs")
    if megalodon_cfg.get("z_dim") != model_dim // 4:
        errors.append("Megalodon z_dim must be model_dim / 4")
    if megalodon_cfg.get("value_dim") != 2 * model_dim:
        errors.append("Megalodon value_dim must be 2 * model_dim")
    if megalodon_cfg.get("ffn_hidden_dim") != 2 * model_dim:
        errors.append("Megalodon ffn_hidden_dim must be 2 * model_dim")
    if megalodon_cfg.get("cema_ndim") != 16:
        errors.append("Megalodon cema_ndim must be 16")
    if megalodon_cfg.get("chunk_size") != megalodon_cfg.get("seq_len"):
        errors.append("Megalodon chunk_size must equal the training sequence length")
    if megalodon_cfg.get("norm_num_groups") != model_dim // 32:
        errors.append("Megalodon norm groups must preserve width 32")
    if megalodon_cfg.get("rope_base") != 100_000.0:
        errors.append("Megalodon RoPE base must be 100,000")
    if megalodon_cfg.get("swiglu") is not True:
        errors.append("Megalodon must use SwiGLU")
    if llama_cfg.get("dim_head") != 128:
        errors.append("Llama must preserve the 128-wide Llama 2 attention head")
    if llama_cfg.get("heads", 0) * llama_cfg.get("dim_head", 0) != llama_dim:
        errors.append("Llama attention heads must span model width")
    if llama_cfg.get("ffn_hidden_dim") != 1024:
        errors.append("The d=384 Llama baseline must use the rounded 8d/3 SwiGLU width")
    if llama_cfg.get("rope_theta") != 10_000.0:
        errors.append("Llama RoPE base must be 10,000")
    if errors:
        raise ValueError("Invalid paper-scaled geometry: " + "; ".join(errors))


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk.

    Args:
        path: YAML file to load.

    Returns:
        The parsed configuration mapping.
    """
    with open(path) as file:
        return yaml.safe_load(file)


def _validate_protocol(megalodon_cfg: dict[str, Any], llama_cfg: dict[str, Any]) -> None:
    """Fail before launching if the declared comparison protocol differs.

    Args:
        megalodon_cfg: Megalodon experiment configuration.
        llama_cfg: Llama experiment configuration.

    Raises:
        ValueError: If model identities, shared settings, or comparison geometry differ.
    """
    if megalodon_cfg.get("model") != "megalodon" or llama_cfg.get("model") != "llama":
        raise ValueError("Expected one megalodon config and one llama config")
    mismatches = {
        key: (megalodon_cfg.get(key), llama_cfg.get(key))
        for key in COMMON_PROTOCOL_KEYS
        if megalodon_cfg.get(key) != llama_cfg.get(key)
    }
    if mismatches:
        raise ValueError(f"Comparison protocol mismatch: {mismatches}")
    if megalodon_cfg.get("share_emb") is not False:
        raise ValueError("The approved comparison requires explicit untied outputs")
    if megalodon_cfg.get("comparison_basis") != "paper_scaled_width_depth":
        raise ValueError("The comparison requires the paper-scaled width/depth basis")
    _validate_paper_scaled_geometry(megalodon_cfg, llama_cfg)


def _is_complete(run_dir: Path, expected_config: dict[str, Any]) -> bool:
    """Check whether a run is complete and matches the requested experiment.

    Args:
        run_dir: Directory containing the run artifacts.
        expected_config: Raw or validated configuration expected for the run. Equivalent
            run-directory path spellings are accepted.

    Returns:
        Whether all required artifacts exist and pass identity and completeness checks.

    Raises:
        ValueError: If the expected configuration is invalid.
    """
    expected_config = validate_config(expected_config.copy())
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    if not all(path.is_file() for path in (summary_path, manifest_path, config_path, metrics_path)):
        return False
    try:
        summary = _read_json(summary_path)
        manifest = _read_json(manifest_path)
        actual_config = _load_yaml(config_path)
        if not isinstance(actual_config, dict):
            return False
        actual_run_dir = actual_config.get("run_dir")
        expected_run_dir = expected_config.get("run_dir")
        if not isinstance(actual_run_dir, str) or not isinstance(expected_run_dir, str):
            return False
        resolved_run_dir = run_dir.resolve()
        if (
            Path(actual_run_dir).resolve() != resolved_run_dir
            or Path(expected_run_dir).resolve() != resolved_run_dir
        ):
            return False
        actual_config["run_dir"] = str(resolved_run_dir)
        expected_config = {**expected_config, "run_dir": str(resolved_run_dir)}
        metrics = _read_jsonl(metrics_path)
        model_file = manifest.get("model_file")
        if not isinstance(model_file, str) or Path(model_file).name != model_file:
            return False
        model_path = run_dir / model_file
        train_steps = [item.get("step") for item in metrics if item.get("kind") == "train"]
        validation_steps = [
            item.get("step") for item in metrics if item.get("kind") == "validation"
        ]
        endpoint = expected_config["num_batches"]
        return (
            actual_config == expected_config
            and summary.get("model") == expected_config["model"]
            and summary.get("seed") == expected_config["seed"]
            and summary.get("completed_steps") == endpoint
            and manifest.get("model_type") == expected_config["model"]
            and manifest.get("seed") == expected_config["seed"]
            and manifest.get("training_step") == endpoint
            and train_steps == list(range(1, endpoint + 1))
            and bool(validation_steps)
            and validation_steps[0] == 0
            and validation_steps[-1] == endpoint
            and model_path.is_file()
        )
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError):
        return False


def _run_one(config: Path, run_dir: Path, seed: int) -> None:
    """Launch an isolated training process or reuse an exact completed run.

    Args:
        config: Base experiment configuration file.
        run_dir: Directory in which to store or find the seeded run.
        seed: Random seed for the run.

    Raises:
        FileExistsError: If ``run_dir`` contains an incomplete run.
        subprocess.CalledProcessError: If the training process fails.
    """
    run_dir = run_dir.resolve()
    expected_config = _load_yaml(config)
    expected_config["seed"] = seed
    expected_config["run_dir"] = str(run_dir)
    if _is_complete(run_dir, expected_config):
        print(f"Reusing completed run: {run_dir}", flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Incomplete non-empty run directory: {run_dir}")
    environment = os.environ.copy()
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    command = [
        sys.executable,
        "train.py",
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
        "--seed",
        str(seed),
    ]
    subprocess.run(command, check=True, env=environment)


def _read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Args:
        path: JSON file to load.

    Returns:
        The parsed JSON object.
    """
    with open(path) as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file and reject non-object records.

    Args:
        path: JSON Lines file to load.

    Returns:
        Parsed object records in file order.

    Raises:
        ValueError: If any record is not a JSON object.
    """
    records: list[dict[str, Any]] = []
    with open(path) as file:
        for line_number, line in enumerate(file, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def _mean_std(values: list[float]) -> dict[str, float]:
    """Calculate the mean and sample standard deviation of values.

    Args:
        values: Numeric observations to summarize.

    Returns:
        A mapping containing the mean and sample standard deviation.
    """
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _mean_std_if_available(values: list[float | None]) -> dict[str, float] | None:
    """Summarize an optional metric only when every run reports it.

    Args:
        values: Per-run numeric observations, including unavailable values.

    Returns:
        Mean and sample standard deviation, or ``None`` when any observation is unavailable.
    """
    if not values or any(value is None for value in values):
        return None
    return _mean_std([float(value) for value in values])


def _interpolate_validation_at_flops(
    validation: list[dict[str, Any]], flop_budget: float
) -> dict[str, float]:
    """Linearly interpolate a validation endpoint at an XLA FLOP budget.

    Args:
        validation: Validation trajectory ordered by cumulative XLA FLOPs.
        flop_budget: Cumulative training FLOPs at which to interpolate.

    Returns:
        Interpolated step, token count, loss, BPC, and FLOP coordinate.

    Raises:
        ValueError: If the budget is invalid or outside a usable trajectory.
    """
    if flop_budget < 0:
        raise ValueError(f"flop_budget must be non-negative, got {flop_budget}")
    if not validation or any("xla_estimated_training_flops" not in item for item in validation):
        raise ValueError("Validation trajectory does not contain XLA FLOP coordinates")

    for left, right in zip(validation, validation[1:], strict=False):
        left_flops = float(left["xla_estimated_training_flops"])
        right_flops = float(right["xla_estimated_training_flops"])
        if left_flops <= flop_budget <= right_flops:
            if right_flops == left_flops:
                fraction = 0.0
            else:
                fraction = (flop_budget - left_flops) / (right_flops - left_flops)

            return {
                "xla_estimated_training_flops": flop_budget,
                **{
                    output_key: float(left[input_key])
                    + fraction * (float(right[input_key]) - float(left[input_key]))
                    for output_key, input_key in (
                        ("interpolated_step", "step"),
                        ("interpolated_tokens_seen", "tokens_seen"),
                        ("loss", "loss"),
                        ("bpc", "bpc"),
                    )
                },
            }
    raise ValueError(
        f"FLOP budget {flop_budget} is outside validation trajectory "
        f"[{validation[0]['xla_estimated_training_flops']}, "
        f"{validation[-1]['xla_estimated_training_flops']}]"
    )


def _aggregate(
    output_dir: Path, seeds: tuple[int, ...], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate final primary endpoints and paired seed deltas.

    Args:
        output_dir: Root directory containing per-seed model runs.
        seeds: Paired random seeds in declared execution order.
        protocol: Shared comparison settings recorded with the aggregate.

    Returns:
        Comparison metadata, per-run records, model summaries, and paired deltas.

    Raises:
        ValueError: If run identities, endpoints, or parameter counts are inconsistent.
    """
    runs: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    parameter_counts = {"megalodon": set(), "llama": set()}
    for seed in seeds:
        summaries: dict[str, dict[str, Any]] = {}
        validation_by_model: dict[str, list[dict[str, Any]]] = {}
        for model in ("megalodon", "llama"):
            run_dir = output_dir / f"seed_{seed}" / model
            summary = _read_json(run_dir / "summary.json")
            manifest = _read_json(run_dir / "manifest.json")
            validation = []
            for metric in _read_jsonl(run_dir / "metrics.jsonl"):
                if metric.get("kind") != "validation":
                    continue
                endpoint = {
                    "step": metric["step"],
                    "tokens_seen": metric["tokens_seen"],
                    "loss": metric["loss"],
                    "bpc": metric["bpc"],
                }
                if "xla_estimated_training_flops" in metric:
                    endpoint["xla_estimated_training_flops"] = metric[
                        "xla_estimated_training_flops"
                    ]
                validation.append(endpoint)
            if summary["seed"] != seed or summary["model"] != model:
                raise ValueError(f"Run identity mismatch in {run_dir}")
            summaries[model] = summary
            validation_by_model[model] = validation
            if summary["completed_steps"] != protocol["num_batches"]:
                raise ValueError(f"Incomplete declared endpoint in {run_dir}")
            if not validation or validation[0]["step"] != 0:
                raise ValueError(f"Missing initial validation endpoint in {run_dir}")
            if validation[-1]["step"] != protocol["num_batches"]:
                raise ValueError(f"Missing final validation endpoint in {run_dir}")
            parameter_counts[model].add(int(summary["parameter_count"]))
            runs[f"{model}_seed_{seed}"] = {
                "run_dir": str(run_dir),
                "summary": summary,
                "manifest": manifest,
                "validation": validation,
            }

        mega = summaries["megalodon"]
        llama = summaries["llama"]
        pair = {
            "seed": seed,
            "megalodon_minus_llama": {
                "validation_loss": mega["final_validation_loss"] - llama["final_validation_loss"],
                "validation_bpc": mega["final_validation_bpc"] - llama["final_validation_bpc"],
                "step_seconds_median": mega["steady_state"]["step_seconds_median"]
                - llama["steady_state"]["step_seconds_median"],
            },
        }
        final_flops = [
            summary.get("xla_estimated_training_flops") for summary in summaries.values()
        ]
        if all(value is not None for value in final_flops):
            common_flop_budget = min(float(value) for value in final_flops)
            compute_matched = {
                model: _interpolate_validation_at_flops(
                    validation_by_model[model], common_flop_budget
                )
                for model in ("megalodon", "llama")
            }
            compute_matched["megalodon_minus_llama"] = {
                key: compute_matched["megalodon"][key] - compute_matched["llama"][key]
                for key in ("loss", "bpc")
            }
            pair["compute_matched"] = compute_matched
        pairs.append(pair)

    mega_count = runs[f"megalodon_seed_{seeds[0]}"]["summary"]["parameter_count"]
    llama_count = runs[f"llama_seed_{seeds[0]}"]["summary"]["parameter_count"]
    if any(len(counts) != 1 for counts in parameter_counts.values()):
        raise ValueError(f"Parameter counts changed across seeds: {parameter_counts}")
    relative_gap = abs(mega_count - llama_count) / min(mega_count, llama_count)
    comparison_basis = protocol.get("comparison_basis")
    loss_deltas = [pair["megalodon_minus_llama"]["validation_loss"] for pair in pairs]
    bpc_deltas = [pair["megalodon_minus_llama"]["validation_bpc"] for pair in pairs]
    timing_deltas = [pair["megalodon_minus_llama"]["step_seconds_median"] for pair in pairs]
    compute_matched_pairs = [pair["compute_matched"] for pair in pairs if "compute_matched" in pair]
    model_metrics: dict[str, Any] = {}
    for model in ("megalodon", "llama"):
        model_summaries = [runs[f"{model}_seed_{seed}"]["summary"] for seed in seeds]
        model_metrics[model] = {
            "final_validation_loss": _mean_std(
                [summary["final_validation_loss"] for summary in model_summaries]
            ),
            "final_validation_bpc": _mean_std(
                [summary["final_validation_bpc"] for summary in model_summaries]
            ),
            "step_seconds_median": _mean_std(
                [summary["steady_state"]["step_seconds_median"] for summary in model_summaries]
            ),
            "step_seconds_p90": _mean_std(
                [summary["steady_state"]["step_seconds_p90"] for summary in model_summaries]
            ),
            "tokens_per_second_aggregate": _mean_std(
                [
                    summary["steady_state"]["tokens_per_second_aggregate"]
                    for summary in model_summaries
                ]
            ),
            "cold_compile_train_seconds": _mean_std(
                [summary["cold_compile_seconds"]["train_step"] for summary in model_summaries]
            ),
            "cold_compile_eval_seconds": _mean_std(
                [summary["cold_compile_seconds"]["eval_step"] for summary in model_summaries]
            ),
            "xla_estimated_flops_per_step": _mean_std_if_available(
                [summary.get("xla_estimated_flops_per_step") for summary in model_summaries]
            ),
            "xla_estimated_training_flops": _mean_std_if_available(
                [summary.get("xla_estimated_training_flops") for summary in model_summaries]
            ),
            "xla_device_peak_bytes_estimate": _mean_std(
                [
                    summary.get("xla_compiler_analysis", {})
                    .get("memory_analysis", {})
                    .get("device_peak_bytes_estimate", 0.0)
                    for summary in model_summaries
                ]
            ),
        }

    return {
        "schema_version": 2,
        "primary_endpoint": "final_validation_loss_at_declared_token_budget",
        "seeds": list(seeds),
        "execution_order": [
            {
                "seed": seed,
                "models": ["megalodon", "llama"] if index % 2 == 0 else ["llama", "megalodon"],
            }
            for index, seed in enumerate(seeds)
        ],
        "protocol": protocol,
        "architecture_comparison": {
            "basis": comparison_basis,
            "megalodon": mega_count,
            "llama": llama_count,
            "absolute_gap": abs(mega_count - llama_count),
            "relative_gap": relative_gap,
        },
        "model_metrics": model_metrics,
        "paired_deltas": pairs,
        "paired_delta_summary": {
            "validation_loss": _mean_std(loss_deltas),
            "validation_bpc": _mean_std(bpc_deltas),
            "step_seconds_median": _mean_std(timing_deltas),
        },
        "compute_matched_secondary": (
            {
                "method": "linear_interpolation_between_fixed_validation_points_at_the_smaller_final_xla_estimated_training_flop_budget",
                "paired_validation_loss_delta": _mean_std(
                    [item["megalodon_minus_llama"]["loss"] for item in compute_matched_pairs]
                ),
                "paired_validation_bpc_delta": _mean_std(
                    [item["megalodon_minus_llama"]["bpc"] for item in compute_matched_pairs]
                ),
            }
            if len(compute_matched_pairs) == len(pairs)
            else None
        ),
        "runs": runs,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write a JSON object atomically through a sibling temporary file.

    Args:
        path: Destination JSON file.
        value: JSON-compatible mapping to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with open(temporary, "w") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def main() -> None:
    """Run the paired experiments in alternating order and aggregate them."""
    parser = argparse.ArgumentParser(
        description="Run the paired Megalodon/Llama enwik8 comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--megalodon-config", default="configs/megalodon_paper_scaled_512.yaml")
    parser.add_argument("--llama-config", default="configs/llama2_paper_scaled_512.yaml")
    parser.add_argument("--output-dir", default="runs/paper_scaled_enwik8_1200")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    mega_config_path = Path(args.megalodon_config)
    llama_config_path = Path(args.llama_config)
    configs = {
        "megalodon": validate_config(_load_yaml(mega_config_path)),
        "llama": validate_config(_load_yaml(llama_config_path)),
    }
    _validate_protocol(configs["megalodon"], configs["llama"])
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be a non-empty unique list")

    output_dir = Path(args.output_dir)
    config_paths = {"megalodon": mega_config_path, "llama": llama_config_path}
    for index, seed in enumerate(seeds):
        order = ("megalodon", "llama") if index % 2 == 0 else ("llama", "megalodon")
        for model in order:
            _run_one(config_paths[model], output_dir / f"seed_{seed}" / model, seed)

    protocol = {key: configs["megalodon"].get(key) for key in COMMON_PROTOCOL_KEYS}
    protocol["learning_rates"] = {
        model: configs[model].get("learning_rate") for model in ("megalodon", "llama")
    }
    result = _aggregate(output_dir, seeds, protocol)
    results_path = output_dir / "comparison.json"
    _write_json_atomic(results_path, result)
    print(f"Wrote paired comparison: {results_path}")


if __name__ == "__main__":
    main()
