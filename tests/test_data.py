"""Tests for data loading and sampling."""

from __future__ import annotations

import numpy as np

from megalodon_enwik8_jax.utils import (
    decode_tokens,
    encode_prompt,
    load_enwik8,
    sample_accum_batch,
    sample_batch,
)


class TestLoadEnwik8:
    """Tests for enwik8 data loading."""

    def test_load_returns_uint8(self) -> None:
        """load_enwik8 returns uint8 arrays."""
        train, val = load_enwik8("data/enwik8.gz")
        assert train.dtype == np.uint8
        assert val.dtype == np.uint8

    def test_load_split_sizes(self) -> None:
        """Default 90/10 train/val split."""
        train, val = load_enwik8("data/enwik8.gz")
        total = len(train) + len(val)
        # Should be ~95M bytes by default
        assert total <= 95_000_000
        # Verify rough 90/10 split
        train_ratio = len(train) / total
        assert 0.89 < train_ratio < 0.91

    def test_values_in_byte_range(self) -> None:
        """All values should be valid bytes [0, 255]."""
        train, val = load_enwik8("data/enwik8.gz")
        assert train.min() >= 0
        assert train.max() <= 255
        assert val.min() >= 0
        assert val.max() <= 255


class TestSampleBatch:
    """Tests for batch sampling."""

    def test_sample_batch_shapes(self, rng: np.random.Generator) -> None:
        """sample_batch returns correct shapes."""
        train, _ = load_enwik8("data/enwik8.gz")
        batch_size, seq_len = 4, 128

        input_ids, labels = sample_batch(rng, train, batch_size, seq_len)

        assert input_ids.shape == (batch_size, seq_len)
        assert labels.shape == (batch_size, seq_len)

    def test_sample_batch_dtype(self, rng: np.random.Generator) -> None:
        """sample_batch returns int32 arrays."""
        train, _ = load_enwik8("data/enwik8.gz")
        input_ids, labels = sample_batch(rng, train, 2, 64)

        assert input_ids.dtype == np.int32
        assert labels.dtype == np.int32

    def test_sample_batch_values(self, rng: np.random.Generator) -> None:
        """Values should be in [0, 255] (byte range)."""
        train, _ = load_enwik8("data/enwik8.gz")
        input_ids, labels = sample_batch(rng, train, 4, 128)

        assert input_ids.min() >= 0
        assert input_ids.max() <= 255
        assert labels.min() >= 0
        assert labels.max() <= 255

    def test_labels_shifted_by_one(self, rng: np.random.Generator) -> None:
        """Labels should be input shifted by one position."""
        train, _ = load_enwik8("data/enwik8.gz")
        # Use a small sequence to verify
        input_ids, labels = sample_batch(rng, train, 1, 10)

        # For autoregressive LM, labels[i] should predict the next token
        # labels = input_ids shifted left by 1 for each sequence
        input_np = np.asarray(input_ids)
        labels_np = np.asarray(labels)
        assert np.array_equal(labels_np[:, :-1], input_np[:, 1:])


class TestSampleAccumBatch:
    """Tests for gradient accumulation batch sampling."""

    def test_accum_batch_shapes(self, rng: np.random.Generator) -> None:
        """sample_accum_batch returns [A, B, T] shaped arrays."""
        train, _ = load_enwik8("data/enwik8.gz")
        batch_size, grad_accum, seq_len = 2, 4, 64

        input_ids, labels = sample_accum_batch(rng, train, batch_size, grad_accum, seq_len)

        assert input_ids.shape == (grad_accum, batch_size, seq_len)
        assert labels.shape == (grad_accum, batch_size, seq_len)

    def test_accum_batch_dtype(self, rng: np.random.Generator) -> None:
        """sample_accum_batch returns int32 arrays."""
        train, _ = load_enwik8("data/enwik8.gz")
        input_ids, labels = sample_accum_batch(rng, train, 2, 2, 32)

        assert input_ids.dtype == np.int32
        assert labels.dtype == np.int32


class TestEncodeDecodeTokens:
    """Tests for token encoding/decoding."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encoding then decoding should return original string."""
        text = "Hello, World!"
        tokens = encode_prompt(text)
        decoded = decode_tokens(tokens)
        assert decoded == text

    def test_encode_returns_int32(self) -> None:
        """encode_prompt returns int32 array."""
        tokens = encode_prompt("test")
        assert tokens.dtype == np.int32

    def test_encode_byte_values(self) -> None:
        """Tokens should be byte values."""
        tokens = encode_prompt("ABC")
        # encode_prompt returns shape [1, T], flatten to check values
        assert list(np.asarray(tokens).flatten()) == [65, 66, 67]  # ASCII values

    def test_decode_handles_special_chars(self) -> None:
        """decode_tokens replaces control chars with spaces for display."""
        import jax.numpy as jnp

        # Tab, newline, carriage return (all < 32) are replaced with space
        tokens = jnp.array([9, 10, 13], dtype=jnp.int32)
        decoded = decode_tokens(tokens)
        # Implementation replaces control chars (< 32) with space for display
        assert decoded == "   "
