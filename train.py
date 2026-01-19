#!/usr/bin/env python3
"""Training script for megalodon-enwik8-jax.

Unified training loop for both Llama and Megalodon models on enwik8
character-level language modeling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from megalodon_enwik8_jax.models import build_model
from megalodon_enwik8_jax.utils import (
    TrainState,
    assert_mask_dtype,
    assert_trainable_dtype,
    bpc_from_loss,
    build_optimizer,
    cast_trainable,
    count_trainable_params,
    create_train_state,
    decode_tokens,
    generate,
    get_dtype,
    load_checkpoint,
    load_config,
    load_enwik8,
    make_cast_mask,
    make_eval_step,
    make_train_step,
    make_trainable_mask,
    resolve_run_dir,
    run_validation,
    sample_accum_batch,
    sample_batch,
    sample_trainable_dtypes,
    save_checkpoint,
    validate_config,
)


def _count_params(model: object, trainable_mask: Any) -> int:
    """Count total trainable parameters in an Equinox model."""
    return count_trainable_params(model, trainable_mask)


def _format_params(n: int) -> str:
    """Format parameter count with K/M/B suffix."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _log_metrics(path: Path, metrics: dict) -> None:
    """Append metrics as JSONL to file."""
    with open(path, "a") as f:
        f.write(
            json.dumps({k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()})
            + "\n"
        )


def main() -> None:
    """Run the training script."""
    parser = argparse.ArgumentParser(
        description="Train language model on enwik8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/test.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--run_dir", type=str, default=None, help="Override run directory from config"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    # Load and validate config
    cfg = load_config(args.config)
    cfg = validate_config(cfg)

    if args.seed is not None:
        cfg["seed"] = args.seed

    # Resolve run directory
    run_dir = resolve_run_dir(cfg, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    # Print config summary
    model_type = cfg["model"]
    print(f"Model: {model_type}")
    print(f"Run directory: {run_dir}")
    print(f"Config: {args.config}")

    # Setup random state
    seed = cfg.get("seed", 42)
    key = jax.random.PRNGKey(seed)
    np_rng = np.random.default_rng(seed)

    # Load data
    print("Loading data...")
    train_data, val_data = load_enwik8(cfg["data_path"])
    print(f"Train size: {len(train_data):,} bytes")
    print(f"Val size: {len(val_data):,} bytes")

    # Build model
    print("Building model...")
    key, model_key = jax.random.split(key)
    model = build_model(cfg, model_key)
    trainable_mask = make_trainable_mask(model)
    cast_mask = make_cast_mask(model, trainable_mask)
    dtype = get_dtype(cfg)
    cast_params = dtype != jnp.float32 and cfg["model"] != "llama"
    print(f"Compute dtype: {dtype}")
    if cast_params:
        model = cast_trainable(model, dtype, trainable_mask, cast_mask=cast_mask)
        assert_mask_dtype(model, dtype, cast_mask)
        dtype_samples = ", ".join(str(item) for item in sample_trainable_dtypes(model, cast_mask))
        print(f"Cast dtype samples: {dtype_samples}")
    else:
        assert_trainable_dtype(model, jnp.float32, trainable_mask)
        dtype_samples = ", ".join(
            str(item) for item in sample_trainable_dtypes(model, trainable_mask)
        )
        print(f"Trainable dtype samples: {dtype_samples}")
    n_params = _count_params(model, trainable_mask)
    print(f"Parameters: {_format_params(n_params)} ({n_params:,})")

    # Build optimizer
    optimizer = build_optimizer(cfg)

    # Create training state
    key, state_key = jax.random.split(key)
    state = create_train_state(model, optimizer, state_key, step=0, trainable_mask=trainable_mask)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        state = load_checkpoint(
            args.resume,
            cfg,
            state_key,
            model_builder=build_model,
            optimizer=optimizer,
        )
        if cast_params:
            state = TrainState(
                step=state.step,
                model=cast_trainable(state.model, dtype, trainable_mask, cast_mask=cast_mask),
                opt_state=state.opt_state,
                key=state.key,
            )
            assert_mask_dtype(state.model, dtype, cast_mask)
        print(f"Resumed at step {int(state.step)}")

    # Create JIT-compiled functions
    print("Compiling training functions...")
    train_step = make_train_step(cfg, optimizer, trainable_mask)
    eval_step = make_eval_step(cfg)

    # Training hyperparameters
    num_batches = cfg.get("num_batches", 1200)
    batch_size = cfg.get("batch_size", 1)
    grad_accum = cfg.get("grad_accum_every", 1)
    seq_len = cfg.get("seq_len", 512)
    validate_every = cfg.get("validate_every", 100)
    generate_every = cfg.get("generate_every", 100)
    save_every = cfg.get("save_every", 500)
    generate_prompt_len = cfg.get("generate_prompt_len", 128)
    generate_length = cfg.get("generate_length", 128)

    # Training loop
    pbar = tqdm(range(int(state.step), num_batches), desc="Training")

    for _ in pbar:
        step = int(state.step)

        # Validation
        if step % validate_every == 0:
            val_loss = run_validation(state.model, eval_step, val_data, np_rng, cfg)
            val_bpc = bpc_from_loss(val_loss)
            _log_metrics(
                metrics_path,
                {"step": step, "val_loss": float(val_loss), "val_bpc": float(val_bpc)},
            )
            tqdm.write(f"Step {step} | Val loss: {val_loss:.4f} | Val BPC: {val_bpc:.4f}")

        # Training step
        input_ids, labels = sample_accum_batch(np_rng, train_data, batch_size, grad_accum, seq_len)
        state, metrics = train_step(state, input_ids, labels)

        # Log training metrics
        train_loss = float(metrics["loss"])
        train_bpc = float(bpc_from_loss(metrics["loss"]))
        grad_norm = float(metrics["grad_norm"])
        _log_metrics(
            metrics_path,
            {
                "step": step,
                "train_loss": train_loss,
                "train_bpc": train_bpc,
                "grad_norm": grad_norm,
            },
        )
        pbar.set_postfix({"loss": f"{train_loss:.4f}", "bpc": f"{train_bpc:.4f}"})

        # Generation (skip if generate_every=0)
        if generate_every > 0 and step % generate_every == 0 and step > 0:
            prompt_input, _ = sample_batch(np_rng, val_data, 1, generate_prompt_len)
            key, gen_key = jax.random.split(key)
            generated = generate(
                state.model,
                prompt_input,
                max_new_tokens=generate_length,
                temperature=cfg.get("temperature", 1.0),
                min_p=cfg.get("min_p", 0.1),
                key=gen_key,
            )
            prompt_text = decode_tokens(prompt_input[0])
            gen_text = decode_tokens(generated[0, generate_prompt_len:])
            tqdm.write(f"\n{'=' * 60} Step {step} {'=' * 60}")
            tqdm.write(f"Prompt: {prompt_text}")
            tqdm.write(f"Generated: {gen_text}")

        # Save checkpoint
        if step % save_every == 0 and step > 0:
            ckpt_path = save_checkpoint(run_dir, state, cfg)
            tqdm.write(f"Saved checkpoint: {ckpt_path}")

    # Final validation (always record end-of-run metric)
    final_step = int(state.step)
    val_loss = run_validation(state.model, eval_step, val_data, np_rng, cfg)
    val_bpc = bpc_from_loss(val_loss)
    _log_metrics(
        metrics_path,
        {"step": final_step, "val_loss": float(val_loss), "val_bpc": float(val_bpc)},
    )
    print(f"\nStep {final_step} | Val loss: {val_loss:.4f} | Val BPC: {val_bpc:.4f}")

    # Final checkpoint
    ckpt_path = save_checkpoint(run_dir, state, cfg, tag="final")
    print(f"\nTraining complete! Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
