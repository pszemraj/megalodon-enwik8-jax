#!/usr/bin/env python3
"""Train either comparison model under the shared enwik8 protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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


def _log_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Append one fully synchronized metrics record.

    Args:
        path: JSON Lines file to append to.
        metrics: Metric names and values for one record.
    """
    with open(path, "a") as file:
        json.dump(
            {
                key: value.item() if hasattr(value, "item") else value
                for key, value in metrics.items()
            },
            file,
        )
        file.write("\n")


def _timing_summary(step_seconds: list[float], tokens_per_step: int) -> dict[str, Any]:
    """Summarize synchronized steady-state step timings.

    Args:
        step_seconds: Wall-clock duration of each training step.
        tokens_per_step: Number of input tokens processed per step.

    Returns:
        Median step time and aggregate throughput after warmup exclusion.
    """
    warmup_excluded = min(TIMING_WARMUP_STEPS, max(0, len(step_seconds) - 1))
    steady = np.asarray(step_seconds[warmup_excluded:], dtype=np.float64)
    median = float(np.median(steady))
    return {
        "warmup_steps_excluded": warmup_excluded,
        "measured_steps": int(steady.size),
        "step_seconds_median": median,
        "tokens_per_second": float(tokens_per_step * steady.size / steady.sum()),
    }


def main() -> None:
    """Run one deterministic, model-only training experiment."""
    parser = argparse.ArgumentParser(
        description="Train a comparison model on the local enwik8 dataset",
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
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.touch(exist_ok=False)

    model_type = cfg["model"]
    seed = int(cfg["seed"])
    print(f"Model: {model_type}")
    print(f"Run directory: {run_dir}")
    print(f"Seed: {seed}")

    print("Loading enwik8...")
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
    jax.block_until_ready(compiled_train_step(state, compile_inputs, compile_labels))
    train_compile_seconds = time.perf_counter() - started

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
    _log_metrics(
        metrics_path,
        initial_validation,
    )
    print(f"Step 0 | Val loss: {val_loss:.4f} | Val BPC: {val_bpc:.4f}")

    step_seconds: list[float] = []
    progress = tqdm(range(1, num_steps + 1), desc="Training")
    for _ in progress:
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
            _log_metrics(
                metrics_path,
                validation_metrics,
            )
            tqdm.write(f"Step {completed_step} | Val loss: {val_loss:.4f} | Val BPC: {val_bpc:.4f}")

    timing = _timing_summary(step_seconds, tokens_per_step)
    summary = {
        "model": model_type,
        "seed": seed,
        "completed_steps": int(state.step),
        "tokens_seen": int(state.step) * tokens_per_step,
        "parameter_count": parameter_count,
        "final_validation_loss": val_loss,
        "final_validation_bpc": val_bpc,
        "steady_state": timing,
    }
    model_path = save_model_artifact(run_dir, state.model, cfg)
    with open(run_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    print(f"Training complete: {model_path}")
    print(
        f"Steady-state median {timing['step_seconds_median']:.4f}s/step, "
        f"{timing['tokens_per_second']:.0f} tokens/s"
    )


if __name__ == "__main__":
    main()
