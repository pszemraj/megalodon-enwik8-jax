#!/usr/bin/env python3
"""Inference script for megalodon-enwik8-jax.

Load a checkpoint and generate text from a prompt.

Usage:
    python inference.py --ckpt runs/llama/checkpoint_1000.eqx --prompt "The "
    python inference.py --ckpt runs/megalodon/checkpoint_1000.eqx --prompt "In the " --max_new_tokens 256
"""

from __future__ import annotations

import argparse

import jax

from megalodon_enwik8_jax.checkpoint import load_checkpoint, load_config_from_checkpoint
from megalodon_enwik8_jax.config import validate_config
from megalodon_enwik8_jax.data import decode_tokens, encode_prompt
from megalodon_enwik8_jax.generate import generate
from megalodon_enwik8_jax.models import build_model
from megalodon_enwik8_jax.optim import build_optimizer
from megalodon_enwik8_jax.utils import count_params, format_params


def main():
    """Main inference function."""
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
        "--top_k",
        type=int,
        default=None,
        help="Top-k filtering (None to disable)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) filtering (None to disable)",
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

    n_params = count_params(state.model)
    print(f"Parameters: {format_params(n_params)} ({n_params:,})")
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
        top_k=args.top_k,
        top_p=args.top_p,
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


if __name__ == "__main__":
    main()
