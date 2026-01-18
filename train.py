#!/usr/bin/env python3
"""Training script for megalodon-enwik8-jax.

Unified training loop for both Llama and Megalodon models on enwik8
character-level language modeling.
"""

from __future__ import annotations

from megalodon_enwik8_jax.cli import train_main


def main() -> None:
    """Run the training CLI."""
    train_main()


if __name__ == "__main__":
    main()
