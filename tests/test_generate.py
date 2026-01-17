"""Tests for text generation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from megalodon_enwik8_jax.generate import generate
from megalodon_enwik8_jax.models import build_model
from megalodon_enwik8_jax.sampling import (
    apply_min_p,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_next_token,
)


class TestGenerate:
    """Tests for generation function."""

    def test_generate_returns_correct_shape(self, key, test_config):
        """generate returns [B, T_prompt + max_new_tokens] array."""
        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)

        batch_size, prompt_len = 2, 16
        max_new_tokens = 8
        prompt_ids = jax.random.randint(key, (batch_size, prompt_len), 0, 256)

        key, gen_key = jax.random.split(key)
        generated = generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            key=gen_key,
        )

        assert generated.shape == (batch_size, prompt_len + max_new_tokens)

    def test_generate_preserves_prompt(self, key, test_config):
        """generate preserves the original prompt tokens."""
        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)

        batch_size, prompt_len = 1, 16
        max_new_tokens = 4
        prompt_ids = jax.random.randint(key, (batch_size, prompt_len), 0, 256)

        key, gen_key = jax.random.split(key)
        generated = generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            key=gen_key,
        )

        # First prompt_len tokens should match original
        assert jnp.array_equal(generated[:, :prompt_len], prompt_ids)

    def test_generate_tokens_in_valid_range(self, key, test_config):
        """All generated tokens should be in [0, 255]."""
        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)

        prompt_ids = jax.random.randint(key, (1, 16), 0, 256)

        key, gen_key = jax.random.split(key)
        generated = generate(
            model,
            prompt_ids,
            max_new_tokens=16,
            temperature=1.0,
            key=gen_key,
        )

        assert generated.min() >= 0
        assert generated.max() <= 255

    def test_generate_with_min_p(self, key, test_config):
        """generate works with min_p sampling."""
        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)

        prompt_ids = jax.random.randint(key, (1, 16), 0, 256)

        key, gen_key = jax.random.split(key)
        generated = generate(
            model,
            prompt_ids,
            max_new_tokens=8,
            temperature=1.0,
            min_p=0.1,
            key=gen_key,
        )

        assert generated.shape == (1, 24)

    def test_generate_deterministic_with_same_key(self, key, test_config):
        """generate produces same output with same PRNG key."""
        key, model_key = jax.random.split(key)
        model = build_model(test_config, model_key)

        prompt_ids = jax.random.randint(key, (1, 16), 0, 256)
        gen_key = jax.random.PRNGKey(123)

        generated1 = generate(
            model,
            prompt_ids,
            max_new_tokens=8,
            temperature=1.0,
            key=gen_key,
        )
        generated2 = generate(
            model,
            prompt_ids,
            max_new_tokens=8,
            temperature=1.0,
            key=gen_key,
        )

        assert jnp.array_equal(generated1, generated2)


class TestSamplingPrimitives:
    """Tests for sampling primitive functions."""

    def test_apply_temperature_scaling(self, key):
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

    def test_apply_min_p_masks_low_prob(self, key):
        """min_p masks tokens below threshold."""
        # Logits that give clear probability differences
        logits = jnp.array([[0.0, -10.0, -10.0]])  # First token has ~100% prob

        filtered = apply_min_p(logits, min_p=0.01)

        # Low probability tokens should be masked to -inf
        probs = jax.nn.softmax(logits)
        max_prob = probs.max()
        threshold = 0.01 * max_prob

        # Tokens with prob < threshold should be -inf
        for i, prob in enumerate(probs[0]):
            if prob < threshold:
                assert filtered[0, i] == float("-inf")

    def test_apply_top_k_keeps_k_tokens(self, key):
        """top_k keeps only top k tokens."""
        logits = jnp.array([[5.0, 3.0, 4.0, 1.0, 2.0]])

        filtered = apply_top_k(logits, k=2)

        # Should keep indices 0 and 2 (values 5.0 and 4.0)
        # Others should be -inf
        assert filtered[0, 0] == 5.0
        assert filtered[0, 2] == 4.0
        assert filtered[0, 1] == float("-inf")
        assert filtered[0, 3] == float("-inf")
        assert filtered[0, 4] == float("-inf")

    def test_apply_top_p_nucleus(self, key):
        """top_p implements nucleus sampling."""
        # Create logits with known softmax distribution
        logits = jnp.array([[10.0, 5.0, 0.0, -5.0, -10.0]])

        # With low p, should keep fewer tokens
        filtered_low = apply_top_p(logits, p=0.9)
        filtered_high = apply_top_p(logits, p=0.99)

        # Higher p should keep more tokens (fewer -inf)
        n_kept_low = jnp.sum(filtered_low > float("-inf"))
        n_kept_high = jnp.sum(filtered_high > float("-inf"))
        assert n_kept_high >= n_kept_low

    def test_sample_next_token_valid_output(self, key):
        """sample_next_token returns valid tokens."""
        logits = jnp.array([[1.0, 2.0, 3.0]])

        new_key, tokens = sample_next_token(key, logits)

        # Should return new key and token indices
        assert new_key is not None
        assert tokens.shape == (1,)
        assert 0 <= int(tokens[0]) < 3

    def test_sample_next_token_respects_distribution(self, key):
        """sample_next_token samples according to softmax distribution."""
        # Heavily biased logits - token 2 should be sampled almost always
        logits = jnp.array([[-100.0, -100.0, 100.0]])

        # Sample many times
        samples = []
        current_key = key
        for _ in range(100):
            current_key, token = sample_next_token(current_key, logits)
            samples.append(int(token[0]))

        # Token 2 should dominate
        assert samples.count(2) > 95
