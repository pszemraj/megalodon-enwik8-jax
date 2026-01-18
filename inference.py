#!/usr/bin/env python3
"""Inference script for megalodon-enwik8-jax.

Load a checkpoint and generate text from a prompt.
"""

from __future__ import annotations

from megalodon_enwik8_jax.cli import inference_main


def main() -> None:
    """Run the inference CLI."""
    inference_main()


if __name__ == "__main__":
    main()
