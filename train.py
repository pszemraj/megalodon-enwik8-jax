#!/usr/bin/env python3
"""Train either comparison model under the shared enwik8 protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from megalodon_enwik8_jax.models import build_model
from megalodon_enwik8_jax.utils import (
    assert_trainable_dtype,
    bpc_from_loss,
    build_optimizer,
    count_trainable_params,
    create_train_state,
    get_dtype,
    learning_rate_at_step,
    load_config,
    load_enwik8,
    make_eval_step,
    make_fixed_batches,
    make_train_step,
    make_trainable_mask,
    resolve_run_dir,
    run_validation,
    sample_accum_batch,
    sample_trainable_dtypes,
    save_model_artifact,
    validate_config,
    write_json_atomic,
    write_yaml_atomic,
)

DATA_BYTES_LIMIT = 95_000_000
DATA_TRAIN_SPLIT = 0.9
TIMING_WARMUP_STEPS = 10


def _format_params(count: int) -> str:
    """Format a parameter count with a compact suffix.

    Args:
        count: Number of parameters.

    Returns:
        A compact human-readable parameter count.
    """
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    if count >= 1e6:
        return f"{count / 1e6:.2f}M"
    if count >= 1e3:
        return f"{count / 1e3:.1f}K"
    return str(count)


def _json_value(value: Any) -> Any:
    """Convert a scalar array into a JSON-compatible Python value.

    Args:
        value: Value to convert if it provides a scalar ``item`` method.

    Returns:
        The converted scalar, or the original value when no conversion is needed.
    """
    if hasattr(value, "item"):
        return value.item()
    return value


def _log_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Append one fully synchronized metrics record.

    Args:
        path: JSON Lines file to append to.
        metrics: Metric names and values for one record.
    """
    with open(path, "a") as file:
        json.dump({key: _json_value(value) for key, value in metrics.items()}, file)
        file.write("\n")


def _prepare_run_dir(run_dir: Path) -> None:
    """Create a fresh run directory and refuse ambiguous stale state."""
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Use a fresh directory for a one-shot run."
        )
    run_dir.mkdir(parents=True, exist_ok=True)


def _git_provenance() -> dict[str, Any]:
    """Capture the local source revision without mutating the repository."""

    def run(*args: str) -> str | None:
        """Run a Git query and return its standard output on success.

        Args:
            *args: Arguments passed to the Git command.

        Returns:
            Stripped standard output, or ``None`` when Git exits unsuccessfully.
        """
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _package_versions() -> dict[str, str | None]:
    """Record versions of the load-bearing runtime packages."""
    packages = ("megalodon-enwik8-jax", "megalodon-jax", "jax", "jaxlib", "equinox", "optax")
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _dtype_counts(model: object, trainable_mask: Any) -> dict[str, int]:
    """Count trainable scalar parameters by storage data type.

    Args:
        model: Model pytree containing parameter arrays.
        trainable_mask: Boolean pytree selecting trainable leaves.

    Returns:
        Mapping from dtype names to trainable scalar counts.
    """
    params = eqx.filter(model, trainable_mask)
    counts: dict[str, int] = {}
    for leaf in jax.tree.leaves(params):
        if leaf is None:
            continue
        dtype = str(leaf.dtype)
        counts[dtype] = counts.get(dtype, 0) + leaf.size
    return counts


def _timing_summary(step_seconds: list[float], tokens_per_step: int) -> dict[str, Any]:
    """Summarize synchronized steady-state step timings.

    Args:
        step_seconds: Wall-clock duration of each training step.
        tokens_per_step: Number of input tokens processed per step.

    Returns:
        Timing percentiles and aggregate throughput after warmup exclusion.
    """
    warmup_excluded = min(TIMING_WARMUP_STEPS, max(0, len(step_seconds) - 1))
    steady = np.asarray(step_seconds[warmup_excluded:], dtype=np.float64)
    median = float(np.median(steady))
    return {
        "warmup_steps_excluded": warmup_excluded,
        "measured_steps": int(steady.size),
        "step_seconds_median": median,
        "step_seconds_p90": float(np.percentile(steady, 90)),
        "tokens_per_second_median": float(tokens_per_step / median),
        "tokens_per_second_aggregate": float(tokens_per_step * steady.size / steady.sum()),
    }


def _compiled_train_analysis(compiled_step: Any) -> dict[str, Any]:
    """Extract XLA's static cost and memory estimates from a compiled step.

    Args:
        compiled_step: Lowered training step, optionally carrying an executable.

    Returns:
        Available cost and memory estimates, or an empty mapping when the step
        has no compiled executable.
    """
    executable = getattr(compiled_step, "compiled", None)
    if executable is None:
        return {}

    cost = executable.cost_analysis() or {}
    if isinstance(cost, list):
        cost = cost[0] if len(cost) == 1 else {}

    result: dict[str, Any] = {
        "cost_analysis": {
            key: float(value)
            for key, value in cost.items()
            if key in {"flops", "transcendentals", "bytes accessed"}
            and isinstance(value, (int, float, np.number))
        }
    }
    memory = executable.memory_analysis()
    if memory is None:
        return result

    memory_fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    memory_values = {
        field: int(getattr(memory, field))
        for field in memory_fields
        if getattr(memory, field, None) is not None
    }
    required = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
    )
    if all(field in memory_values for field in required):
        memory_values["device_peak_bytes_estimate"] = (
            memory_values["argument_size_in_bytes"]
            + memory_values["output_size_in_bytes"]
            - memory_values["alias_size_in_bytes"]
            + memory_values["temp_size_in_bytes"]
        )
    result["memory_analysis"] = memory_values
    return result


def main() -> None:
    """Run one deterministic, model-only training experiment."""
    parser = argparse.ArgumentParser(
        description="Train a comparison model on the checked-in enwik8 data contract",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/test.yaml", help="YAML experiment config")
    parser.add_argument("--run-dir", "--run_dir", dest="run_dir", help="Fresh output directory")
    parser.add_argument("--seed", type=int, help="Override the config seed")
    args = parser.parse_args()

    cfg = validate_config(load_config(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    run_dir = resolve_run_dir(cfg, args.run_dir)
    cfg["run_dir"] = str(run_dir)
    _prepare_run_dir(run_dir)
    write_yaml_atomic(run_dir / "config.yaml", cfg)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.touch(exist_ok=False)

    model_type = cfg["model"]
    seed = int(cfg["seed"])
    print(f"Model: {model_type}")
    print(f"Run directory: {run_dir}")
    print(f"Seed: {seed}")

    print("Loading the local enwik8 data contract...")
    train_data, val_data = load_enwik8(
        cfg["data_path"],
        bytes_limit=DATA_BYTES_LIMIT,
        train_split=DATA_TRAIN_SPLIT,
    )
    print(f"Train size: {len(train_data):,} bytes")
    print(f"Validation size: {len(val_data):,} bytes")

    key = jax.random.PRNGKey(seed)
    key, model_key, state_key = jax.random.split(key, 3)
    model = build_model(cfg, model_key)
    trainable_mask = make_trainable_mask(model)
    if model_type == "llama":
        assert_trainable_dtype(model, jnp.float32, trainable_mask)
    parameter_count = count_trainable_params(model, trainable_mask)
    dtype_samples = ", ".join(str(item) for item in sample_trainable_dtypes(model, trainable_mask))
    print(f"Compute dtype: {get_dtype(cfg)}")
    print(f"Trainable dtype samples: {dtype_samples}")
    print(f"Parameters: {_format_params(parameter_count)} ({parameter_count:,})")

    optimizer = build_optimizer(cfg)
    state = create_train_state(
        model,
        optimizer,
        state_key,
        step=0,
        trainable_mask=trainable_mask,
    )
    train_step = make_train_step(cfg, optimizer, trainable_mask)
    eval_step = make_eval_step(cfg)

    num_steps = int(cfg.get("num_batches", 1200))
    batch_size = int(cfg["batch_size"])
    grad_accum = int(cfg["grad_accum_every"])
    seq_len = int(cfg.get("seq_len", 512))
    validate_every = int(cfg["validate_every"])
    val_batch_size = int(cfg["val_batch_size"])
    val_batches_count = int(cfg["val_batches"])
    if num_steps <= 0 or validate_every <= 0:
        raise ValueError("num_batches and validate_every must be positive")
    tokens_per_step = batch_size * grad_accum * seq_len

    validation_batches = make_fixed_batches(
        val_data,
        val_batch_size,
        val_batches_count,
        seq_len,
    )
    train_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    compile_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    compile_inputs, compile_labels = sample_accum_batch(
        compile_rng,
        train_data,
        batch_size,
        grad_accum,
        seq_len,
    )

    print("Compiling and synchronizing the training step...")
    started = time.perf_counter()
    if cfg["jit"]:
        compiled_train_step = train_step.lower(state, compile_inputs, compile_labels).compile()
    else:
        compiled_train_step = train_step
    compiled_state, compiled_metrics = compiled_train_step(state, compile_inputs, compile_labels)
    jax.block_until_ready((compiled_state, compiled_metrics))
    train_compile_seconds = time.perf_counter() - started
    compiler_analysis = _compiled_train_analysis(compiled_train_step)
    xla_flops_per_step = compiler_analysis.get("cost_analysis", {}).get("flops")

    print("Compiling and synchronizing the evaluation step...")
    started = time.perf_counter()
    if cfg["jit"]:
        compiled_eval_step = eval_step.lower(
            state.model,
            validation_batches[0][0],
            validation_batches[1][0],
        ).compile()
    else:
        compiled_eval_step = eval_step
    compiled_eval = compiled_eval_step(
        state.model,
        validation_batches[0][0],
        validation_batches[1][0],
    )
    compiled_eval.block_until_ready()
    eval_compile_seconds = time.perf_counter() - started
    print(f"Cold compile: train {train_compile_seconds:.3f}s, eval {eval_compile_seconds:.3f}s")

    val_loss = float(run_validation(state.model, compiled_eval_step, validation_batches))
    val_bpc = float(bpc_from_loss(jnp.asarray(val_loss)))
    initial_validation = {
        "kind": "validation",
        "step": 0,
        "tokens_seen": 0,
        "loss": val_loss,
        "bpc": val_bpc,
    }
    if xla_flops_per_step is not None:
        initial_validation["xla_estimated_training_flops"] = 0.0
    _log_metrics(
        metrics_path,
        initial_validation,
    )
    print(f"Step 0 | Val loss: {val_loss:.4f} | Val BPC: {val_bpc:.4f}")

    step_seconds: list[float] = []
    progress = tqdm(range(1, num_steps + 1), desc="Training")
    for expected_step in progress:
        input_ids, labels = sample_accum_batch(
            train_rng,
            train_data,
            batch_size,
            grad_accum,
            seq_len,
        )
        started = time.perf_counter()
        state, metrics = compiled_train_step(state, input_ids, labels)
        jax.block_until_ready((state, metrics))
        elapsed = time.perf_counter() - started
        step_seconds.append(elapsed)

        completed_step = int(state.step)
        if completed_step != expected_step:
            raise RuntimeError(
                f"Training step mismatch: expected {expected_step}, got {completed_step}"
            )
        train_loss = float(metrics["loss"])
        train_bpc = float(bpc_from_loss(metrics["loss"]))
        # Reconstructing the logged rate is reporting-only and occurs after synchronized timing.
        train_metrics = {
            "kind": "train",
            "step": completed_step,
            "tokens_seen": completed_step * tokens_per_step,
            "loss": train_loss,
            "bpc": train_bpc,
            "grad_norm": float(metrics["grad_norm"]),
            "learning_rate": learning_rate_at_step(cfg, completed_step - 1),
            "step_seconds": elapsed,
        }
        if xla_flops_per_step is not None:
            train_metrics["xla_estimated_training_flops"] = completed_step * xla_flops_per_step
        _log_metrics(metrics_path, train_metrics)
        progress.set_postfix(loss=f"{train_loss:.4f}", bpc=f"{train_bpc:.4f}")

        if completed_step % validate_every == 0 or completed_step == num_steps:
            val_loss = float(run_validation(state.model, compiled_eval_step, validation_batches))
            val_bpc = float(bpc_from_loss(jnp.asarray(val_loss)))
            validation_metrics = {
                "kind": "validation",
                "step": completed_step,
                "tokens_seen": completed_step * tokens_per_step,
                "loss": val_loss,
                "bpc": val_bpc,
            }
            if xla_flops_per_step is not None:
                validation_metrics["xla_estimated_training_flops"] = (
                    completed_step * xla_flops_per_step
                )
            _log_metrics(
                metrics_path,
                validation_metrics,
            )
            tqdm.write(f"Step {completed_step} | Val loss: {val_loss:.4f} | Val BPC: {val_bpc:.4f}")

    timing = _timing_summary(step_seconds, tokens_per_step)
    summary = {
        "schema_version": 1,
        "model": model_type,
        "seed": seed,
        "completed_steps": int(state.step),
        "tokens_seen": int(state.step) * tokens_per_step,
        "parameter_count": parameter_count,
        "final_validation_loss": val_loss,
        "final_validation_bpc": val_bpc,
        "cold_compile_seconds": {
            "train_step": train_compile_seconds,
            "eval_step": eval_compile_seconds,
        },
        "xla_compiler_analysis": compiler_analysis,
        "xla_estimated_flops_per_step": xla_flops_per_step,
        "xla_estimated_training_flops": (
            int(state.step) * xla_flops_per_step if xla_flops_per_step is not None else None
        ),
        "steady_state": timing,
    }
    manifest_metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "training_step": int(state.step),
        "seed": seed,
        "parameter_count": parameter_count,
        "trainable_dtype_counts": _dtype_counts(state.model, trainable_mask),
        "packages": _package_versions(),
        "git": _git_provenance(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "data": {
            "path": str(cfg["data_path"]),
            "bytes_limit": DATA_BYTES_LIMIT,
            "train_split": DATA_TRAIN_SPLIT,
            "train_bytes": len(train_data),
            "validation_bytes": len(val_data),
            "validation_windows": "evenly_spaced_fixed",
            "validation_window_count": val_batch_size * val_batches_count,
            "validation_batch_size": val_batch_size,
            "validation_batch_count": val_batches_count,
        },
    }
    model_path = save_model_artifact(run_dir, state.model, cfg, manifest_metadata)
    write_json_atomic(run_dir / "summary.json", summary)
    print(f"Training complete: {model_path}")
    print(
        f"Steady-state median {timing['step_seconds_median']:.4f}s/step, "
        f"{timing['tokens_per_second_aggregate']:.0f} tokens/s"
    )


if __name__ == "__main__":
    main()
