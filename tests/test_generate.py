"""Tests for text generation."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from megalodon_enwik8_jax.models import build_model
from megalodon_enwik8_jax.utils import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    generate,
    sample_token,
)


class TestGenerate:
    """Tests for generation function."""

    def test_generate_returns_correct_shape(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
    ) -> None:
        """generate returns [B, T_prompt + max_new_tokens] array."""
        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)

        batch_size, prompt_len = 2, 16
        max_new_tokens = 8
        prompt_ids = jax.random.randint(key, (batch_size, prompt_len), 0, 256)

        key, gen_key = jax.random.split(key)
        generated, artifact, next_key = generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            key=gen_key,
        )

        assert generated.shape == (batch_size, prompt_len + max_new_tokens)
        assert artifact is not None
        assert next_key is not None

    def test_generate_preserves_prompt(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
    ) -> None:
        """generate preserves the original prompt tokens."""
        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)

        batch_size, prompt_len = 1, 16
        max_new_tokens = 4
        prompt_ids = jax.random.randint(key, (batch_size, prompt_len), 0, 256)

        key, gen_key = jax.random.split(key)
        generated, _, _ = generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            key=gen_key,
        )

        # First prompt_len tokens should match original
        assert jnp.array_equal(generated[:, :prompt_len], prompt_ids)

    def test_generate_tokens_in_valid_range(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
    ) -> None:
        """All generated tokens should be in [0, 255]."""
        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)

        prompt_ids = jax.random.randint(key, (1, 16), 0, 256)

        key, gen_key = jax.random.split(key)
        generated, _, _ = generate(
            model,
            prompt_ids,
            max_new_tokens=16,
            temperature=1.0,
            key=gen_key,
        )

        assert generated.min() >= 0
        assert generated.max() <= 255

    def test_generate_with_top_k_and_top_p(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
    ) -> None:
        """generate supports the sampling controls shared by both models."""
        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)

        prompt_ids = jax.random.randint(key, (1, 16), 0, 256)

        key, gen_key = jax.random.split(key)
        generated, _, _ = generate(
            model,
            prompt_ids,
            max_new_tokens=8,
            temperature=1.0,
            top_k=32,
            top_p=0.9,
            key=gen_key,
        )

        assert generated.shape == (1, 24)

    def test_generate_deterministic_with_same_key(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
    ) -> None:
        """generate produces same output with same PRNG key."""
        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)

        prompt_ids = jax.random.randint(key, (1, 16), 0, 256)
        gen_key = jax.random.PRNGKey(123)

        generated1, _, next_key1 = generate(
            model,
            prompt_ids,
            max_new_tokens=8,
            temperature=1.0,
            key=gen_key,
        )
        generated2, _, next_key2 = generate(
            model,
            prompt_ids,
            max_new_tokens=8,
            temperature=1.0,
            key=gen_key,
        )

        assert jnp.array_equal(generated1, generated2)
        assert jnp.array_equal(next_key1, next_key2)

    def test_generate_rejects_llama_sequence_beyond_rope_capacity(
        self,
        key: jax.Array,
        test_config: dict[str, Any],
    ) -> None:
        """Llama generation fails clearly before overrunning its RoPE table."""
        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)
        prompt_ids = jnp.zeros((1, model.config.max_seq_len), dtype=jnp.int32)

        with pytest.raises(
            ValueError,
            match=r"generation length 66 exceeds the RoPE capacity of 64",
        ):
            generate(
                model,
                prompt_ids,
                max_new_tokens=2,
                key=key,
            )


class TestSamplingPrimitives:
    """Tests for sampling primitive functions."""

    def test_apply_temperature_scaling(self, key: jax.Array) -> None:
        """Temperature scales logits correctly."""
        logits = jnp.array([[1.0, 2.0, 3.0]])

        # Temperature 1.0 should not change logits
        scaled = apply_temperature(logits, 1.0)
        assert jnp.allclose(scaled, logits)

        # Temperature 2.0 should halve logits
        scaled = apply_temperature(logits, 2.0)
        assert jnp.allclose(scaled, logits / 2.0)

        # Temperature 0.5 should double logits
        scaled = apply_temperature(logits, 0.5)
        assert jnp.allclose(scaled, logits * 2.0)

    def test_apply_top_k_keeps_only_requested_logits(self) -> None:
        """Top-k filtering retains exactly k logits, including across ties."""
        logits = jnp.array(
            [
                [1.0, 4.0, 3.0, 2.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )

        filtered = apply_top_k(logits, top_k=2)

        assert jnp.array_equal(jnp.isfinite(filtered).sum(axis=-1), jnp.array([2, 2]))
        assert jnp.isfinite(filtered[0, 1])
        assert jnp.isfinite(filtered[0, 2])
        assert jnp.isfinite(filtered[1, 0])
        assert jnp.isfinite(filtered[1, 1])

    def test_apply_top_p_keeps_crossing_token(self) -> None:
        """Nucleus filtering never drops the token that crosses top_p."""
        logits = jnp.log(jnp.array([[0.5, 0.3, 0.15, 0.05]]))

        filtered = apply_top_p(logits, top_p=0.7)

        assert jnp.isfinite(filtered[0, 0])
        assert jnp.isfinite(filtered[0, 1])
        assert not jnp.isfinite(filtered[0, 2])
        assert not jnp.isfinite(filtered[0, 3])

    def test_sample_token_valid_output(self, key: jax.Array) -> None:
        """sample_token returns valid tokens."""
        logits = jnp.array([[1.0, 2.0, 3.0]])

        new_key, tokens = sample_token(key, logits)

        # Should return new key and token indices
        assert new_key is not None
        assert tokens.shape == (1,)
        assert 0 <= int(tokens[0]) < 3

    def test_sample_token_respects_distribution(self, key: jax.Array) -> None:
        """sample_token samples according to softmax distribution."""
        # Heavily biased logits - token 2 should be sampled almost always
        logits = jnp.array([[-100.0, -100.0, 100.0]])

        # Sample many times
        samples = []
        current_key = key
        for _ in range(100):
            current_key, token = sample_token(current_key, logits)
            samples.append(int(token[0]))

        # Token 2 should dominate
        assert samples.count(2) > 95
