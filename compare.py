#!/usr/bin/env python3
"""Run and summarize paired Megalodon/Llama experiments."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_SEEDS = (7, 17, 42)


def _run_one(config: Path, run_dir: Path, seed: int) -> None:
    """Launch one training run in a separate process.

    Args:
        config: Experiment configuration file.
        run_dir: Output directory for the run.
        seed: Training seed.
    """
    environment = os.environ.copy()
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    command = [
        sys.executable,
        "train.py",
        "--config",
        str(config),
        "--run-dir",
        str(run_dir.resolve()),
        "--seed",
        str(seed),
    ]
    subprocess.run(command, check=True, env=environment)


def _mean_std(values: list[float]) -> dict[str, float]:
    """Calculate a mean and sample standard deviation.

    Args:
        values: Numeric observations.

    Returns:
        Mean and sample standard deviation.
    """
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _aggregate(output_dir: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    """Summarize final metrics and paired seed deltas.

    Args:
        output_dir: Root directory containing per-seed runs.
        seeds: Paired seeds in execution order.

    Returns:
        Per-seed results and aggregate comparison metrics.
    """
    summaries: dict[str, list[dict[str, Any]]] = {"megalodon": [], "llama": []}
    pairs: list[dict[str, Any]] = []

    for seed in seeds:
        pair: dict[str, Any] = {"seed": seed}
        for model in ("megalodon", "llama"):
            with open(output_dir / f"seed_{seed}" / model / "summary.json") as file:
                summary = json.load(file)
            summaries[model].append(summary)
            pair[model] = {
                "validation_loss": summary["final_validation_loss"],
                "validation_bpc": summary["final_validation_bpc"],
                "step_seconds": summary["steady_state"]["step_seconds_median"],
                "tokens_per_second": summary["steady_state"]["tokens_per_second"],
            }

        pair["megalodon_minus_llama"] = {
            key: pair["megalodon"][key] - pair["llama"][key]
            for key in ("validation_loss", "validation_bpc", "step_seconds")
        }
        pairs.append(pair)

    model_metrics: dict[str, Any] = {}
    for model, model_summaries in summaries.items():
        model_metrics[model] = {
            "validation_loss": _mean_std(
                [float(summary["final_validation_loss"]) for summary in model_summaries]
            ),
            "validation_bpc": _mean_std(
                [float(summary["final_validation_bpc"]) for summary in model_summaries]
            ),
            "step_seconds": _mean_std(
                [
                    float(summary["steady_state"]["step_seconds_median"])
                    for summary in model_summaries
                ]
            ),
            "tokens_per_second": _mean_std(
                [float(summary["steady_state"]["tokens_per_second"]) for summary in model_summaries]
            ),
        }

    return {
        "seeds": list(seeds),
        "parameter_counts": {
            model: int(model_summaries[0]["parameter_count"])
            for model, model_summaries in summaries.items()
        },
        "model_metrics": model_metrics,
        "paired_results": pairs,
        "paired_delta_summary": {
            metric: _mean_std([pair["megalodon_minus_llama"][metric] for pair in pairs])
            for metric in ("validation_loss", "validation_bpc", "step_seconds")
        },
    }


def main() -> None:
    """Run paired experiments in alternating order and write their summary."""
    parser = argparse.ArgumentParser(
        description="Run the paired Megalodon/Llama enwik8 comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--megalodon-config", default="configs/megalodon_paper_scaled_512.yaml")
    parser.add_argument("--llama-config", default="configs/llama2_paper_scaled_512.yaml")
    parser.add_argument("--output-dir", default="runs/paper_scaled_enwik8_1200")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild comparison.json from existing per-run summaries",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = {
        "megalodon": Path(args.megalodon_config),
        "llama": Path(args.llama_config),
    }
    seeds = tuple(args.seeds)
    if not args.aggregate_only:
        for index, seed in enumerate(seeds):
            order = ("megalodon", "llama") if index % 2 == 0 else ("llama", "megalodon")
            for model in order:
                _run_one(config_paths[model], output_dir / f"seed_{seed}" / model, seed)

    result = _aggregate(output_dir, seeds)
    results_path = output_dir / "comparison.json"
    with open(results_path, "w") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(f"Wrote paired comparison: {results_path}")


if __name__ == "__main__":
    main()
