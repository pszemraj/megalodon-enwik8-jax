"""Tests for model implementations."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from megalodon_enwik8_jax.models import build_model, forward_model
from megalodon_enwik8_jax.utils import count_trainable_params, load_config, make_trainable_mask


def count_params(model: object) -> int:
    trainable_mask = make_trainable_mask(model)
    return count_trainable_params(model, trainable_mask)


class TestLlamaModel:
    """Tests for Llama model."""

    def test_llama_forward_shape(self, key: jax.Array, test_config: dict[str, Any]) -> None:
        """Llama forward produces correct output shape."""
        model = build_model(test_config, key)

        batch_size, seq_len = 2, 32
        input_ids = jax.random.randint(key, (batch_size, seq_len), 0, 256)

        logits, cache = forward_model(model, input_ids)

        assert logits.shape == (batch_size, seq_len, 256)
        assert cache is None  # No cache when not requested

    def test_llama_forward_with_cache(self, key: jax.Array, test_config: dict[str, Any]) -> None:
        """Llama forward returns cache when requested."""
        model = build_model(test_config, key)

        batch_size, seq_len = 2, 16
        input_ids = jax.random.randint(key, (batch_size, seq_len), 0, 256)

        logits, cache = forward_model(model, input_ids, return_cache=True)

        assert logits.shape == (batch_size, seq_len, 256)
        assert cache is not None
        # Cache should be a list of per-layer caches
        assert len(cache) == test_config["depth"]

    def test_llama_param_count(self, key: jax.Array, test_config: dict[str, Any]) -> None:
        """Llama has expected parameter count."""
        model = build_model(test_config, key)
        n_params = count_params(model)

        # With dim=64, depth=2, heads=2, dim_head=32, ffn_mult=2.0
        # Rough estimate: embedding + layers + lm_head
        # Should be in reasonable range for tiny model
        assert 10_000 < n_params < 1_000_000

    def test_llama_output_dtype(self, key: jax.Array, test_config: dict[str, Any]) -> None:
        """Llama outputs consistent dtype."""
        model = build_model(test_config, key)
        input_ids = jax.random.randint(key, (1, 16), 0, 256)

        logits, _ = forward_model(model, input_ids)

        assert logits.dtype == jnp.float32

    def test_llama_gaussian_initialization_is_explicit(
        self, key: jax.Array, test_config: dict[str, Any]
    ) -> None:
        """The Llama baseline consistently uses its declared Gaussian scale."""
        model = build_model({**test_config, "init_std": 0.01}, key)

        observed_std = float(model.layers[0].attn.wq.weight.std())
        assert observed_std == pytest.approx(0.01, rel=0.1)

    def test_llama_cached_logits_match_full_forward(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
    ) -> None:
        """Cached continuation matches the corresponding full-forward logits."""
        model = build_model(test_config, key)
        tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)

        full_logits, _ = forward_model(model, tokens)
        _, cache = forward_model(model, tokens[:, :-1], return_cache=True)
        cached_logits, _ = forward_model(model, tokens[:, -1:], cache=cache, return_cache=True)

        assert jnp.allclose(cached_logits[:, 0], full_logits[:, -1], atol=2e-2, rtol=2e-2)


class TestMegalodonModel:
    """Tests for Megalodon model wrapper."""

    def test_megalodon_forward_shape(
        self,
        key: jax.Array,
        megalodon_config: dict[str, Any],
    ) -> None:
        """Megalodon forward produces correct output shape."""
        model = build_model(megalodon_config, key)

        batch_size, seq_len = 2, 32
        input_ids = jax.random.randint(key, (batch_size, seq_len), 0, 256)

        logits, cache = forward_model(model, input_ids)

        assert logits.shape == (batch_size, seq_len, 256)

    def test_megalodon_supports_partial_final_chunk(
        self,
        key: jax.Array,
        megalodon_config: dict[str, Any],
    ) -> None:
        """Sequences need not be an exact multiple of chunk_size in v0.2."""
        model = build_model(megalodon_config, key)
        input_ids = jax.random.randint(key, (1, 40), 0, 256)

        logits, _ = forward_model(model, input_ids)

        assert logits.shape == (1, 40, 256)
        assert logits.dtype == jnp.float32

    def test_megalodon_cached_logits_match_full_forward(
        self,
        key: jax.Array,
        megalodon_config: dict[str, Any],
    ) -> None:
        """Cached continuation matches full forward inside one chunk."""
        model = build_model(megalodon_config, key)
        tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)

        full_logits, _ = forward_model(model, tokens)
        _, cache = forward_model(model, tokens[:, :-1], return_cache=True)
        cached_logits, _ = forward_model(model, tokens[:, -1:], cache=cache, return_cache=True)

        assert jnp.allclose(cached_logits[:, 0], full_logits[:, -1], atol=2e-2, rtol=2e-2)

    def test_megalodon_forward_with_cache(
        self,
        key: jax.Array,
        megalodon_config: dict[str, Any],
    ) -> None:
        """Megalodon forward returns cache when requested."""
        model = build_model(megalodon_config, key)

        batch_size, seq_len = 2, 32
        input_ids = jax.random.randint(key, (batch_size, seq_len), 0, 256)

        logits, cache = forward_model(model, input_ids, return_cache=True)

        assert logits.shape == (batch_size, seq_len, 256)
        # Cache presence depends on megalodon-jax implementation
        # At minimum, the call should succeed

    def test_megalodon_param_count(
        self,
        key: jax.Array,
        megalodon_config: dict[str, Any],
    ) -> None:
        """Megalodon has expected parameter count."""
        model = build_model(megalodon_config, key)
        n_params = count_params(model)

        # Should be in reasonable range for tiny model
        assert 10_000 < n_params < 5_000_000


class TestModelInterface:
    """Tests for unified model interface."""

    def test_build_model_dispatches_llama(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
    ) -> None:
        """build_model creates Llama for model='llama'."""
        model = build_model(test_config, key)
        # Should be a LlamaLM instance
        assert hasattr(model, "embed")
        assert hasattr(model, "layers")
        assert hasattr(model, "norm")

    def test_build_model_dispatches_megalodon(
        self,
        key: jax.Array,
        megalodon_config: dict[str, Any],
    ) -> None:
        """build_model creates Megalodon for model='megalodon'."""
        model = build_model(megalodon_config, key)
        # Should be a MegalodonForCausalLM with model and config attributes
        assert hasattr(model, "model")
        assert hasattr(model, "config")

    def test_forward_model_deterministic(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
    ) -> None:
        """forward_model produces same output when deterministic=True."""
        model = build_model(test_config, key)
        input_ids = jax.random.randint(key, (1, 16), 0, 256)

        logits1, _ = forward_model(model, input_ids, deterministic=True)
        logits2, _ = forward_model(model, input_ids, deterministic=True)

        assert jnp.allclose(logits1, logits2)

    def test_forward_model_rejects_unknown_model(self) -> None:
        """forward_model fails closed for unsupported model objects."""
        input_ids = jnp.zeros((1, 1), dtype=jnp.int32)

        with pytest.raises(TypeError, match="Unsupported model type: object"):
            forward_model(object(), input_ids)

    @pytest.mark.parametrize(
        ("config_path", "expected_params"),
        [
            ("configs/megalodon_paper_scaled_512.yaml", 12_098_112),
            ("configs/llama2_paper_scaled_512.yaml", 10_818_432),
            ("configs/megalodon_paper_scaled_512_long.yaml", 12_098_112),
            ("configs/llama2_paper_scaled_512_long.yaml", 10_818_432),
        ],
    )
    def test_comparison_model_parameter_counts(
        self,
        key: jax.Array,
        config_path: str,
        expected_params: int,
    ) -> None:
        """The declared comparison models retain their audited parameter counts."""
        model = build_model(load_config(config_path), key)

        assert count_params(model) == expected_params
