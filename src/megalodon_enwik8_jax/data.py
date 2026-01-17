"""Data loading and batching for enwik8 character-level modeling."""

from __future__ import annotations

import gzip
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def load_enwik8(
    path: str | Path,
    bytes_limit: int = 95_000_000,
    train_split: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Load enwik8 data from gzipped file.

    Reads the first `bytes_limit` bytes of enwik8 for faster iteration.
    The default of 95M (vs full 100M) reduces load time while keeping
    sufficient data for training.

    Args:
        path: Path to enwik8.gz file.
        bytes_limit: Maximum bytes to read (default 95M).
        train_split: Fraction of data for training (default 0.9).

    Returns:
        Tuple of (train_data, val_data) as uint8 numpy arrays.

    Raises:
        FileNotFoundError: If data file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with gzip.open(path) as f:
        data = np.frombuffer(f.read(bytes_limit), dtype=np.uint8).copy()

    train_size = int(len(data) * train_split)
    train_data = data[:train_size]
    val_data = data[train_size:]

    return train_data, val_data


def sample_batch(
    rng: np.random.Generator,
    data_u8: np.ndarray,
    batch_size: int,
    seq_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Sample a random batch of sequences from data.

    Returns (input_ids, labels) where labels are input_ids shifted by 1.
    Each sequence is seq_len+1 tokens, split into seq_len inputs and
    seq_len labels.

    Args:
        rng: Numpy random generator.
        data_u8: Source data as uint8 array.
        batch_size: Number of sequences in batch.
        seq_len: Sequence length (excluding the extra token for labels).

    Returns:
        Tuple of (input_ids, labels), each shape [B, T] as int32.
    """
    max_start = len(data_u8) - seq_len - 1

    # Sample random start positions
    starts = rng.integers(0, max_start, size=(batch_size,))

    # Extract sequences
    sequences = np.stack(
        [data_u8[start : start + seq_len + 1] for start in starts],
        axis=0,
    )

    # Split into inputs and labels
    input_ids = sequences[:, :-1].astype(np.int32)
    labels = sequences[:, 1:].astype(np.int32)

    return jnp.asarray(input_ids), jnp.asarray(labels)


def sample_accum_batch(
    rng: np.random.Generator,
    data_u8: np.ndarray,
    batch_size: int,
    grad_accum: int,
    seq_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Sample a batch for gradient accumulation.

    Returns tensors with an extra leading dimension for accumulation steps.

    Args:
        rng: Numpy random generator.
        data_u8: Source data as uint8 array.
        batch_size: Micro-batch size.
        grad_accum: Number of gradient accumulation steps.
        seq_len: Sequence length.

    Returns:
        Tuple of (input_ids, labels), each shape [A, B, T] as int32,
        where A=grad_accum, B=batch_size, T=seq_len.
    """
    max_start = len(data_u8) - seq_len - 1
    total_seqs = batch_size * grad_accum

    # Sample all sequences at once
    starts = rng.integers(0, max_start, size=(total_seqs,))
    sequences = np.stack(
        [data_u8[start : start + seq_len + 1] for start in starts],
        axis=0,
    )

    # Split into inputs and labels
    input_ids = sequences[:, :-1].astype(np.int32)
    labels = sequences[:, 1:].astype(np.int32)

    # Reshape for accumulation: [A*B, T] -> [A, B, T]
    input_ids = input_ids.reshape(grad_accum, batch_size, seq_len)
    labels = labels.reshape(grad_accum, batch_size, seq_len)

    return jnp.asarray(input_ids), jnp.asarray(labels)


def encode_prompt(text: str) -> jax.Array:
    """Encode text string to token IDs (bytes).

    Args:
        text: Input text string.

    Returns:
        Token IDs as int32 array of shape [1, T].
    """
    tokens = np.array(list(text.encode("utf-8")), dtype=np.int32)
    return jnp.asarray(tokens[None, :])


def decode_tokens(tokens: jax.Array) -> str:
    """Decode token IDs to text string.

    Args:
        tokens: Token IDs as array of shape [T] or [1, T].

    Returns:
        Decoded text string, with invalid bytes replaced.
    """
    tokens = np.asarray(tokens).flatten()
    # Replace control characters with space for display
    bytes_array = bytes(max(32, int(t)) if t < 128 else int(t) for t in tokens)
    return bytes_array.decode("utf-8", errors="replace")
