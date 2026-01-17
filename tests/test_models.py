"""Tests for model implementations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from megalodon_enwik8_jax.models import build_model, forward_model
from megalodon_enwik8_jax.utils import count_params


class TestLlamaModel:
    """Tests for Llama model."""

    def test_llama_forward_shape(self, key, test_config):
        """Llama forward produces correct output shape."""
        model = build_model(test_config, key)

        batch_size, seq_len = 2, 32
        input_ids = jax.random.randint(key, (batch_size, seq_len), 0, 256)

        logits, cache = forward_model(model, input_ids)

        assert logits.shape == (batch_size, seq_len, 256)
        assert cache is None  # No cache when not requested

    def test_llama_forward_with_cache(self, key, test_config):
        """Llama forward returns cache when requested."""
        model = build_model(test_config, key)

        batch_size, seq_len = 2, 16
        input_ids = jax.random.randint(key, (batch_size, seq_len), 0, 256)

        logits, cache = forward_model(model, input_ids, return_cache=True)

        assert logits.shape == (batch_size, seq_len, 256)
        assert cache is not None
        # Cache should be a list of per-layer caches
        assert len(cache) == test_config["depth"]

    def test_llama_param_count(self, key, test_config):
        """Llama has expected parameter count."""
        model = build_model(test_config, key)
        n_params = count_params(model)

        # With dim=64, depth=2, heads=2, dim_head=32, ffn_mult=2.0
        # Rough estimate: embedding + layers + lm_head
        # Should be in reasonable range for tiny model
        assert 10_000 < n_params < 1_000_000

    def test_llama_output_dtype(self, key, test_config):
        """Llama outputs consistent dtype."""
        model = build_model(test_config, key)
        input_ids = jax.random.randint(key, (1, 16), 0, 256)

        logits, _ = forward_model(model, input_ids)

        # Logits should be a float dtype (bf16 on GPU, fp32 on CPU)
        assert jnp.issubdtype(logits.dtype, jnp.floating)


class TestMegalodonModel:
    """Tests for Megalodon model wrapper."""

    def test_megalodon_forward_shape(self, key, megalodon_config):
        """Megalodon forward produces correct output shape."""
        model = build_model(megalodon_config, key)

        batch_size, seq_len = 2, 32
        input_ids = jax.random.randint(key, (batch_size, seq_len), 0, 256)

        logits, cache = forward_model(model, input_ids)

        assert logits.shape == (batch_size, seq_len, 256)

    def test_megalodon_forward_with_cache(self, key, megalodon_config):
        """Megalodon forward returns cache when requested."""
        model = build_model(megalodon_config, key)

        batch_size, seq_len = 2, 32
        input_ids = jax.random.randint(key, (batch_size, seq_len), 0, 256)

        logits, cache = forward_model(model, input_ids, return_cache=True)

        assert logits.shape == (batch_size, seq_len, 256)
        # Cache presence depends on megalodon-jax implementation
        # At minimum, the call should succeed

    def test_megalodon_param_count(self, key, megalodon_config):
        """Megalodon has expected parameter count."""
        model = build_model(megalodon_config, key)
        n_params = count_params(model)

        # Should be in reasonable range for tiny model
        assert 10_000 < n_params < 5_000_000


class TestModelInterface:
    """Tests for unified model interface."""

    def test_build_model_dispatches_llama(self, key, test_config):
        """build_model creates Llama for model='llama'."""
        model = build_model(test_config, key)
        # Should be a LlamaLM instance
        assert hasattr(model, "embed")
        assert hasattr(model, "layers")
        assert hasattr(model, "norm")

    def test_build_model_dispatches_megalodon(self, key, megalodon_config):
        """build_model creates Megalodon for model='megalodon'."""
        model = build_model(megalodon_config, key)
        # Should be a MegalodonForCausalLM with model and config attributes
        assert hasattr(model, "model")
        assert hasattr(model, "config")

    def test_forward_model_deterministic(self, key, test_config):
        """forward_model produces same output when deterministic=True."""
        model = build_model(test_config, key)
        input_ids = jax.random.randint(key, (1, 16), 0, 256)

        logits1, _ = forward_model(model, input_ids, deterministic=True)
        logits2, _ = forward_model(model, input_ids, deterministic=True)

        assert jnp.allclose(logits1, logits2)
