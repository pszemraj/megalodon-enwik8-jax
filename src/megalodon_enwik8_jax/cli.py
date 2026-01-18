"""CLI entrypoints for training and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
from tqdm.auto import tqdm

from .checkpoint import load_checkpoint, load_config_from_checkpoint, save_checkpoint
from .config import get_dtype, load_config, resolve_run_dir, validate_config
from .data import decode_tokens, encode_prompt, load_enwik8, sample_accum_batch, sample_batch
from .generate import generate
from .models import build_model
from .params import (
    assert_trainable_dtype,
    cast_trainable,
    count_trainable_params,
    make_trainable_mask,
    sample_trainable_dtypes,
)
from .training import (
    TrainState,
    bpc_from_loss,
    build_optimizer,
    create_train_state,
    make_eval_step,
    make_train_step,
    run_validation,
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


def train_main() -> None:
    """Run the training CLI."""
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
    dtype = get_dtype(cfg)
    model = cast_trainable(model, dtype, trainable_mask)
    assert_trainable_dtype(model, dtype, trainable_mask)
    dtype_samples = ", ".join(str(item) for item in sample_trainable_dtypes(model, trainable_mask))
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
        state = TrainState(
            step=state.step,
            model=cast_trainable(state.model, dtype, trainable_mask),
            opt_state=state.opt_state,
            key=state.key,
        )
        assert_trainable_dtype(state.model, dtype, trainable_mask)
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

    # Final checkpoint
    ckpt_path = save_checkpoint(run_dir, state, cfg, tag="final")
    print(f"\nTraining complete! Final checkpoint: {ckpt_path}")


def inference_main() -> None:
    """Run the inference CLI."""
    parser = argparse.ArgumentParser(
        description="Generate text from checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The ",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.1,
        help="Min-p filtering threshold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Load config from checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    cfg = load_config_from_checkpoint(args.ckpt)
    cfg = validate_config(cfg)

    model_type = cfg["model"]
    print(f"Model type: {model_type}")

    # Setup random state
    key = jax.random.PRNGKey(args.seed)

    # Build model skeleton and optimizer for loading
    key, model_key = jax.random.split(key)
    optimizer = build_optimizer(cfg)

    # Load checkpoint
    state = load_checkpoint(
        args.ckpt,
        cfg,
        model_key,
        model_builder=build_model,
        optimizer=optimizer,
    )
    dtype = get_dtype(cfg)
    trainable_mask = make_trainable_mask(state.model)
    state = TrainState(
        step=state.step,
        model=cast_trainable(state.model, dtype, trainable_mask),
        opt_state=state.opt_state,
        key=state.key,
    )
    assert_trainable_dtype(state.model, dtype, trainable_mask)

    n_params = _count_params(state.model, trainable_mask)
    print(f"Parameters: {_format_params(n_params)} ({n_params:,})")
    print(f"Checkpoint step: {int(state.step)}")

    # Encode prompt
    prompt_ids = encode_prompt(args.prompt)
    print(f"\nPrompt ({len(args.prompt)} chars): {args.prompt!r}")

    # Generate
    print(f"Generating {args.max_new_tokens} tokens...")
    key, gen_key = jax.random.split(key)

    generated = generate(
        state.model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        min_p=args.min_p,
        key=gen_key,
    )

    # Decode
    prompt_len = prompt_ids.shape[1]
    gen_tokens = generated[0, prompt_len:]
    gen_text = decode_tokens(gen_tokens)

    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt!r}")
    print(f"Generated: {gen_text!r}")
    print("=" * 60)

    # Also print full text
    full_text = args.prompt + gen_text
    print(f"\nFull output:\n{full_text}")


__all__ = ["train_main", "inference_main"]
