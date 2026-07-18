"""Utility helpers for config, data, sampling, training, and checkpoints."""

from __future__ import annotations

import gzip
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jaxtyping import Array, Float, Int

from .models import build_model, forward_model
from .models.llama import LlamaLM

if TYPE_CHECKING:
    from jax.typing import DTypeLike

# =============================================================================
# Config utilities
# =============================================================================

# Known config keys for validation - reject unknown keys to catch typos
KNOWN_KEYS = frozenset(
    {
        # General
        "run_dir",
        "model",
        "seed",
        # Model - shared
        "num_tokens",
        "param_dtype",
        "compute_dtype",
        "accum_dtype",
        "attention_softmax_dtype",
        "loss_softmax_dtype",
        "jit",
        # Llama-specific
        "dim",
        "depth",
        "heads",
        "dim_head",
        "rope_theta",
        "init_std",
        # Megalodon-specific
        "model_dim",
        "num_layers",
        "num_heads",
        "z_dim",
        "value_dim",
        "ffn_hidden_dim",
        "cema_ndim",
        "chunk_size",
        "norm_num_groups",
        "swiglu",
        "rescale_nffn",
        "scale_emb",
        "share_emb",
        "init_mode",
        "rope_base",
        "attention_dropout",
        "hidden_dropout",
        "dropout",
        "use_checkpoint",
        # Training
        "num_batches",
        "batch_size",
        "grad_accum_every",
        "learning_rate",
        "lr_schedule",
        "warmup_steps",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "weight_decay",
        "grad_clip_norm",
        # Data
        "data_path",
        "seq_len",
        # Evaluation
        "validate_every",
        "val_batch_size",
        "val_batches",
    }
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        path: Path to the configuration file.

    Returns:
        The parsed configuration mapping.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    return cfg


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate configuration and apply defaults.

    Args:
        cfg: Configuration mapping to validate and normalize in place.

    Returns:
        The validated configuration mapping with defaults applied.
    """
    # Check for unknown keys
    unknown_keys = set(cfg.keys()) - KNOWN_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {unknown_keys}")

    # Validate model type
    model = cfg.get("model", "llama").lower()
    if model not in {"megalodon", "llama"}:
        raise ValueError(f"model must be 'megalodon' or 'llama', got '{model}'")
    cfg["model"] = model

    # Validate vocab_size
    num_tokens = cfg.get("num_tokens", 256)
    if num_tokens != 256:
        raise ValueError(f"num_tokens must be 256 for enwik8 (bytes), got {num_tokens}")

    precision_keys = (
        "param_dtype",
        "compute_dtype",
        "accum_dtype",
        "attention_softmax_dtype",
        "loss_softmax_dtype",
    )
    for key in precision_keys:
        if key not in cfg:
            continue
        value = cfg[key]
        if value is None:
            continue
        if isinstance(value, str):
            val = value.lower()
            if val in {"bf16", "bfloat16"}:
                cfg[key] = "bf16"
            elif val in {"fp32", "float32"}:
                cfg[key] = "fp32"
            elif val in {"fp16", "float16"}:
                raise ValueError(f"{key} must be bf16/fp32 (no fp16), got '{value}'")
            else:
                raise ValueError(f"{key} must be bf16/fp32, got '{value}'")
        elif isinstance(value, (jnp.dtype, np.dtype)) or value in {jnp.bfloat16, jnp.float32}:
            continue
        else:
            raise ValueError(f"{key} must be bf16/fp32 or a JAX dtype, got '{value}'")

    # Llama baseline does not implement dropout; require zeroed values.
    if model == "llama":
        for key in ("dropout", "attention_dropout", "hidden_dropout"):
            if cfg.get(key, 0.0) > 0.0:
                raise ValueError(f"{key} must be 0.0 for Llama baseline.")

    # Apply defaults for optional fields
    defaults = {
        "seed": 42,
        "jit": True,
        "num_batches": 1200,
        "batch_size": 1,
        "grad_accum_every": 1,
        "learning_rate": 1e-3,
        "lr_schedule": "constant",
        "warmup_steps": 0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "validate_every": 100,
        "val_batches": 10,
    }

    for key, default in defaults.items():
        if key not in cfg:
            cfg[key] = default

    cfg.setdefault("val_batch_size", cfg["batch_size"])

    return cfg


def resolve_run_dir(cfg: dict[str, Any], override: str | None = None) -> Path:
    """Resolve the run directory from the configuration or an override.

    Args:
        cfg: Configuration containing an optional ``run_dir`` value.
        override: Run directory to use instead of the configured value.

    Returns:
        The resolved run-directory path.
    """
    if override:
        return Path(override)
    return Path(cfg.get("run_dir", "runs/default"))


def get_dtype(cfg: dict[str, Any]) -> DTypeLike:
    """Get the compute dtype from a configuration.

    Args:
        cfg: Configuration containing an optional ``compute_dtype`` value.

    Returns:
        The corresponding JAX dtype.
    """
    dtype_value = cfg.get("compute_dtype", "bf16")
    if isinstance(dtype_value, str):
        dtype_str = dtype_value.lower()
        return jnp.bfloat16 if dtype_str in {"bf16", "bfloat16"} else jnp.float32
    if isinstance(dtype_value, (jnp.dtype, np.dtype)):
        return jnp.dtype(dtype_value)
    if dtype_value in {jnp.bfloat16, jnp.float32}:
        return jnp.dtype(dtype_value)
    raise ValueError(f"compute_dtype must be bf16/fp32 or a JAX dtype, got '{dtype_value}'")


# =============================================================================
# Data utilities
# =============================================================================


def load_enwik8(
    path: str | Path,
    bytes_limit: int = 95_000_000,
    train_split: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and split enwik8 data from a gzip file.

    Args:
        path: Path to the compressed enwik8 data.
        bytes_limit: Maximum number of bytes to read.
        train_split: Fraction of loaded bytes assigned to training.

    Returns:
        The training and validation byte arrays.
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
    """Sample a random next-byte prediction batch.

    Args:
        rng: NumPy random-number generator used to choose start offsets.
        data_u8: One-dimensional byte array to sample.
        batch_size: Number of sequences in the batch.
        seq_len: Number of input tokens per sequence.

    Returns:
        The input token IDs and their one-byte-shifted labels.
    """
    max_start = len(data_u8) - seq_len - 1
    starts = rng.integers(0, max_start + 1, size=(batch_size,))
    sequences = np.stack(
        [data_u8[start : start + seq_len + 1] for start in starts],
        axis=0,
    )

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
    """Sample a next-byte prediction batch for gradient accumulation.

    Args:
        rng: NumPy random-number generator used to choose start offsets.
        data_u8: One-dimensional byte array to sample.
        batch_size: Number of sequences in each microbatch.
        grad_accum: Number of microbatches to sample.
        seq_len: Number of input tokens per sequence.

    Returns:
        Input token IDs and shifted labels shaped by accumulation step and batch.
    """
    max_start = len(data_u8) - seq_len - 1
    total_seqs = batch_size * grad_accum
    starts = rng.integers(0, max_start + 1, size=(total_seqs,))
    sequences = np.stack(
        [data_u8[start : start + seq_len + 1] for start in starts],
        axis=0,
    )

    input_ids = sequences[:, :-1].astype(np.int32)
    labels = sequences[:, 1:].astype(np.int32)

    input_ids = input_ids.reshape(grad_accum, batch_size, seq_len)
    labels = labels.reshape(grad_accum, batch_size, seq_len)

    return jnp.asarray(input_ids), jnp.asarray(labels)


def make_fixed_batches(
    data_u8: np.ndarray,
    batch_size: int,
    num_batches: int,
    seq_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Build deterministic, evenly spaced evaluation windows.

    Args:
        data_u8: One-dimensional byte array to sample.
        batch_size: Number of sequences in each evaluation batch.
        num_batches: Number of evaluation batches.
        seq_len: Number of input tokens per sequence.

    Returns:
        Batched input token IDs and their one-byte-shifted labels.
    """
    max_start = len(data_u8) - seq_len - 1
    total_sequences = batch_size * num_batches
    starts = np.linspace(0, max_start, num=total_sequences, dtype=np.int64)
    sequences = np.stack(
        [data_u8[start : start + seq_len + 1] for start in starts],
        axis=0,
    )
    input_ids = sequences[:, :-1].astype(np.int32)
    labels = sequences[:, 1:].astype(np.int32)
    return (
        jnp.asarray(input_ids.reshape(num_batches, batch_size, seq_len)),
        jnp.asarray(labels.reshape(num_batches, batch_size, seq_len)),
    )


def encode_prompt(text: str) -> jax.Array:
    """Encode text as a batch of UTF-8 byte token IDs.

    Args:
        text: Text to encode.

    Returns:
        An integer token array with a leading batch dimension.
    """
    tokens = np.array(list(text.encode("utf-8")), dtype=np.int32)
    return jnp.asarray(tokens[None, :])


def decode_tokens(tokens: jax.Array) -> str:
    """Decode byte token IDs as UTF-8 text.

    Args:
        tokens: Byte-valued token IDs to decode.

    Returns:
        Decoded text, replacing invalid UTF-8 sequences.
    """
    tokens = np.asarray(tokens).flatten()
    bytes_array = bytes(int(t) for t in tokens)
    return bytes_array.decode("utf-8", errors="replace")


# =============================================================================
# Sampling + generation
# =============================================================================


def apply_temperature(logits: jax.Array, temperature: float) -> jax.Array:
    """Apply temperature scaling to logits.

    Args:
        logits: Unnormalized token scores.
        temperature: Positive sampling temperature.

    Returns:
        Temperature-scaled logits.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return logits / temperature


def apply_top_k(logits: jax.Array, top_k: int | None) -> jax.Array:
    """Keep only the top-k logits along the vocabulary axis.

    Args:
        logits: Unnormalized token scores.
        top_k: Number of highest-scoring tokens to retain, or ``None``.

    Returns:
        Filtered logits with excluded values set to negative infinity.
    """
    if top_k is None:
        return logits
    if not 0 < top_k <= logits.shape[-1]:
        raise ValueError(f"top_k must be in [1, {logits.shape[-1]}], got {top_k}")
    top_values, top_indices = jax.lax.top_k(logits, top_k)
    return jnp.put_along_axis(
        jnp.full_like(logits, -jnp.inf),
        top_indices,
        top_values,
        axis=-1,
        inplace=False,
    )


def apply_top_p(logits: jax.Array, top_p: float | None) -> jax.Array:
    """Apply nucleus filtering while retaining the token crossing the threshold.

    Args:
        logits: Unnormalized token scores.
        top_p: Cumulative probability threshold, or ``None``.

    Returns:
        Filtered logits in their original vocabulary order.
    """
    if top_p is None:
        return logits
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_p == 1.0:
        return logits

    order = jnp.argsort(logits, axis=-1)[..., ::-1]
    sorted_logits = jnp.take_along_axis(logits, order, axis=-1)
    sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
    cumulative_before = jnp.cumsum(sorted_probs, axis=-1) - sorted_probs
    sorted_logits = jnp.where(cumulative_before >= top_p, -jnp.inf, sorted_logits)
    inverse_order = jnp.argsort(order, axis=-1)
    return jnp.take_along_axis(sorted_logits, inverse_order, axis=-1)


def sample_token(
    key: jax.Array,
    logits: jax.Array,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Sample a token using temperature, top-k, and nucleus filtering.

    Args:
        key: JAX pseudo-random number generator key.
        logits: Unnormalized token scores.
        temperature: Positive sampling temperature.
        top_k: Number of highest-scoring tokens to retain, or ``None``.
        top_p: Cumulative probability threshold, or ``None``.

    Returns:
        The updated random key and sampled token IDs.
    """
    key, sample_key = jax.random.split(key)
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    tokens = jax.random.categorical(sample_key, logits, axis=-1)
    return key, tokens


def _generate_llama(
    model: LlamaLM,
    prompt_ids: Int[Array, "batch seq"],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    key: jax.Array,
) -> tuple[
    Int[Array, "batch total_seq"],
    list[tuple[Array, Array]] | None,
    jax.Array,
]:
    """Generate Llama continuations with a Python decode loop.

    This demo-oriented path favors a clear eager loop over a compiled, preallocated
    cache and is intended for short qualitative samples rather than throughput work.

    Args:
        model: Llama language model to evaluate.
        prompt_ids: Batched prompt token IDs.
        max_new_tokens: Maximum number of continuation tokens.
        temperature: Positive sampling temperature.
        top_k: Number of highest-scoring tokens to retain, or ``None``.
        top_p: Cumulative probability threshold, or ``None``.
        key: JAX pseudo-random number generator key.

    Returns:
        Generated sequences, the final attention cache, and the updated random key.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if max_new_tokens == 0:
        return prompt_ids, None, key

    requested_seq_len = prompt_ids.shape[1] + max_new_tokens
    if requested_seq_len > model.config.max_seq_len:
        raise ValueError(
            f"Requested Llama generation length {requested_seq_len} exceeds the "
            f"RoPE capacity of {model.config.max_seq_len}; shorten the prompt or "
            "reduce max_new_tokens"
        )

    logits, cache = forward_model(model, prompt_ids, return_cache=True, deterministic=True)

    last_logits = logits[:, -1, :]
    key, next_token = sample_token(key, last_logits, temperature, top_k, top_p)
    next_token = next_token[:, None]

    generated = [next_token]

    for _ in range(max_new_tokens - 1):
        logits, cache = forward_model(
            model, next_token, cache=cache, return_cache=True, deterministic=True
        )
        last_logits = logits[:, -1, :]
        key, next_token = sample_token(key, last_logits, temperature, top_k, top_p)
        next_token = next_token[:, None]
        generated.append(next_token)

    generated = jnp.concatenate(generated, axis=1)
    return jnp.concatenate([prompt_ids, generated], axis=1), cache, key


def generate(
    model: eqx.Module,
    prompt_ids: Int[Array, "batch seq"],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    key: jax.Array | None = None,
) -> tuple[Int[Array, "batch total_seq"], Any, jax.Array | None]:
    """Generate tokens and return the continuation artifact and next random key.

    Args:
        model: Language model to evaluate.
        prompt_ids: Batched prompt token IDs.
        max_new_tokens: Maximum number of continuation tokens.
        temperature: Positive sampling temperature.
        top_k: Number of highest-scoring tokens to retain, or ``None``.
        top_p: Cumulative probability threshold, or ``None``.
        key: JAX pseudo-random number generator key, or ``None`` for a fixed key.

    Returns:
        Generated sequences, the model-specific continuation cache, and the updated key.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    if isinstance(model, LlamaLM):
        return _generate_llama(
            model,
            prompt_ids,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            key,
        )

    from megalodon_jax import generate as megalodon_generate

    return megalodon_generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        key=key,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_cache=True,
    )


# =============================================================================
# Params utilities
# =============================================================================


def make_trainable_mask(model: eqx.Module) -> Any:
    """Create a pytree mask of trainable parameters.

    Args:
        model: Model whose inexact array leaves should be marked trainable.

    Returns:
        A Boolean pytree matching the model structure.
    """
    return jax.tree_util.tree_map(eqx.is_inexact_array, model)


def assert_mask_dtype(model: eqx.Module, dtype: jnp.dtype, mask: Any) -> None:
    """Assert all masked floating-point parameters have the requested dtype.

    Args:
        model: Model whose parameters are inspected.
        dtype: Expected floating-point dtype.
        mask: Pytree selecting the parameters to inspect.
    """
    params = eqx.filter(model, mask)
    leaves = [leaf for leaf in jax.tree.leaves(params) if leaf is not None]

    mismatched = [
        leaf.dtype
        for leaf in leaves
        if jnp.issubdtype(leaf.dtype, jnp.floating) and leaf.dtype != dtype
    ]
    if mismatched:
        unique = sorted({str(item) for item in mismatched})
        raise ValueError(f"Masked params dtype mismatch. Expected {dtype}, found {unique}.")


def assert_trainable_dtype(
    model: eqx.Module,
    dtype: jnp.dtype,
    trainable_mask: Any | None = None,
) -> None:
    """Assert all trainable floating-point parameters have the requested dtype.

    Args:
        model: Model whose parameters are inspected.
        dtype: Expected floating-point dtype.
        trainable_mask: Trainable-parameter mask, or ``None`` to derive one.
    """
    if trainable_mask is None:
        trainable_mask = make_trainable_mask(model)

    assert_mask_dtype(model, dtype, trainable_mask)


def count_trainable_params(
    model: eqx.Module,
    trainable_mask: Any | None = None,
) -> int:
    """Count trainable parameters while excluding non-trainable buffers.

    Args:
        model: Model whose parameters are counted.
        trainable_mask: Trainable-parameter mask, or ``None`` to derive one.

    Returns:
        The number of scalar trainable parameters.
    """
    if trainable_mask is None:
        trainable_mask = make_trainable_mask(model)

    params = eqx.filter(model, trainable_mask)
    return sum(leaf.size for leaf in jax.tree.leaves(params) if leaf is not None)


def sample_trainable_dtypes(
    model: eqx.Module,
    trainable_mask: Any | None = None,
    max_samples: int = 3,
) -> list[jnp.dtype]:
    """Sample representative trainable parameter dtypes for logging.

    Args:
        model: Model whose trainable parameter dtypes are sampled.
        trainable_mask: Trainable-parameter mask, or ``None`` to derive one.
        max_samples: Maximum number of dtype entries to return.

    Returns:
        Up to ``max_samples`` trainable parameter dtypes in pytree order.
    """
    if trainable_mask is None:
        trainable_mask = make_trainable_mask(model)

    params = eqx.filter(model, trainable_mask)
    samples: list[jnp.dtype] = []
    for leaf in jax.tree.leaves(params):
        if leaf is None:
            continue
        samples.append(leaf.dtype)
        if len(samples) >= max_samples:
            break
    return samples


# =============================================================================
# Training utilities
# =============================================================================


EvalStep = Callable[
    [eqx.Module, Int[Array, "batch seq"], Int[Array, "batch seq"]],
    Float[Array, ""],
]


class TrainState(eqx.Module):
    """Container for all mutable training state."""

    step: jax.Array
    model: eqx.Module
    opt_state: optax.OptState
    key: jax.Array


def create_train_state(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    key: jax.Array,
    step: int = 0,
    trainable_mask: Any | None = None,
) -> TrainState:
    """Create initial training state.

    Args:
        model: Model whose trainable leaves initialize the optimizer state.
        optimizer: Optax transformation used for parameter updates.
        key: Initial training PRNG key.
        step: Initial optimizer step.
        trainable_mask: Optional PyTree selecting trainable model leaves.

    Returns:
        Initialized model, optimizer, step, and PRNG state.
    """
    if trainable_mask is None:
        trainable_mask = make_trainable_mask(model)

    params, _ = eqx.partition(model, trainable_mask)
    opt_state = optimizer.init(params)

    return TrainState(
        step=jnp.array(step, dtype=jnp.int32),
        model=model,
        opt_state=opt_state,
        key=key,
    )


LearningRate = float | Callable[[jax.Array], jax.Array]


def build_learning_rate(cfg: dict[str, Any]) -> LearningRate:
    """Build the declared constant or warmup-cosine learning-rate policy.

    Args:
        cfg: Validated experiment configuration.

    Returns:
        A constant learning rate or an Optax schedule callable.
    """
    learning_rate = float(cfg.get("learning_rate", 1e-3))
    schedule = cfg.get("lr_schedule", "constant")
    if schedule == "constant":
        return learning_rate
    if schedule != "warmup_cosine":
        raise ValueError(f"Unsupported lr_schedule: {schedule!r}")

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=int(cfg["warmup_steps"]),
        decay_steps=int(cfg["num_batches"]),
        end_value=0.0,
    )


def learning_rate_at_step(cfg: dict[str, Any], step: int) -> float:
    """Return the rate used by the optimizer update at a zero-based step.

    Args:
        cfg: Validated experiment configuration.
        step: Zero-based optimizer step.

    Returns:
        Learning rate applied at ``step``.
    """
    learning_rate = build_learning_rate(cfg)
    if callable(learning_rate):
        return float(learning_rate(jnp.asarray(step, dtype=jnp.int32)))
    return float(learning_rate)


def build_optimizer(cfg: dict[str, Any]) -> optax.GradientTransformation:
    """Build AdamW with clipping applied to raw gradients.

    Args:
        cfg: Validated experiment configuration.

    Returns:
        Composed Optax gradient transformation.
    """
    learning_rate = build_learning_rate(cfg)
    weight_decay = float(cfg.get("weight_decay", 0.0))
    grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
    b1 = float(cfg.get("adam_beta1", 0.9))
    b2 = float(cfg.get("adam_beta2", 0.999))
    eps = float(cfg.get("adam_eps", 1e-8))

    transforms = []

    if grad_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip_norm))

    transforms.append(
        optax.adamw(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
        )
    )

    return optax.chain(*transforms)


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute mean cross-entropy loss for language modeling.

    Args:
        logits: Model logits with shape ``[batch, sequence, vocabulary]``.
        labels: Target token IDs with shape ``[batch, sequence]``.

    Returns:
        Scalar mean loss over all target tokens.
    """
    return cross_entropy_loss_sum(logits, labels) / labels.size


def cross_entropy_loss_sum(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute the FP32 summed cross-entropy over all target tokens.

    Args:
        logits: Model logits with shape ``[batch, sequence, vocabulary]``.
        labels: Target token IDs with shape ``[batch, sequence]``.

    Returns:
        Scalar FP32 loss sum.
    """
    logits_f32 = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)

    batch_size, seq_len, _ = logits.shape
    batch_idx = jnp.arange(batch_size)[:, None]
    seq_idx = jnp.arange(seq_len)[None, :]
    target_log_probs = log_probs[batch_idx, seq_idx, labels]

    return -target_log_probs.sum(dtype=jnp.float32)


def bpc_from_loss(loss: jax.Array) -> jax.Array:
    """Convert cross-entropy loss to bits-per-character.

    Args:
        loss: Cross-entropy measured in nats.

    Returns:
        Loss measured in bits per character.
    """
    return loss / jnp.log(2.0)


def make_train_step(
    cfg: dict[str, Any],
    optimizer: optax.GradientTransformation,
    trainable_mask: Any,
) -> Callable[
    [TrainState, Int[Array, "accum batch seq"], Int[Array, "accum batch seq"]],
    tuple[TrainState, dict[str, Array]],
]:
    """Create the optionally JIT-compiled training step.

    Args:
        cfg: Validated experiment configuration.
        optimizer: Optax transformation for trainable parameters.
        trainable_mask: PyTree selecting trainable model leaves.

    Returns:
        Function that consumes one accumulated batch and updates the training state.
    """
    use_jit = cfg.get("jit", True)
    dropout_enabled = any(
        cfg.get(key, 0.0) > 0.0 for key in ("dropout", "attention_dropout", "hidden_dropout")
    )
    deterministic = not dropout_enabled

    def train_step(
        state: TrainState,
        input_ids: Int[Array, "accum batch seq"],
        labels: Int[Array, "accum batch seq"],
    ) -> tuple[TrainState, dict[str, Array]]:
        """Apply one optimizer update over an accumulated batch.

        Args:
            state: Current model, optimizer, step, and PRNG state.
            input_ids: Input token microbatches.
            labels: Target token microbatches.

        Returns:
            Updated training state and scalar training metrics.
        """
        params, static = eqx.partition(state.model, trainable_mask)

        def loss_fn(
            params: eqx.Module,
            batch_input: Int[Array, "batch seq"],
            batch_labels: Int[Array, "batch seq"],
            key: jax.Array,
        ) -> Float[Array, ""]:
            """Return the summed token loss for one microbatch.

            Args:
                params: Trainable model leaves.
                batch_input: Input token IDs for one microbatch.
                batch_labels: Target token IDs for one microbatch.
                key: PRNG key for stochastic model operations.

            Returns:
                Scalar summed cross-entropy.
            """
            model = eqx.combine(params, static)
            logits, _ = forward_model(model, batch_input, deterministic=deterministic, key=key)
            return cross_entropy_loss_sum(logits, batch_labels)

        def _add_trees(left: Any, right: Any) -> Any:
            """Add two gradient leaves while preserving static ``None`` leaves.

            Args:
                left: Accumulated gradient leaf or ``None``.
                right: New gradient leaf or ``None``.

            Returns:
                Summed gradient leaf or ``None``.
            """
            if left is None:
                return None
            return left + right

        def _scale_tree(value: Any, scale: float) -> Any:
            """Scale a gradient leaf while preserving static ``None`` leaves.

            Args:
                value: Gradient leaf or ``None``.
                scale: Scalar multiplier.

            Returns:
                Scaled gradient leaf or ``None``.
            """
            if value is None:
                return None
            return value * scale

        def micro_step(
            carry: tuple[jax.Array, Any, jax.Array],
            micro_batch: tuple[Int[Array, "batch seq"], Int[Array, "batch seq"]],
        ) -> tuple[tuple[jax.Array, Any, jax.Array], None]:
            """Accumulate loss and gradients for one scan microbatch.

            Args:
                carry: Current PRNG key, gradient sum, and loss sum.
                micro_batch: Input and target arrays for one microbatch.

            Returns:
                Updated scan carry and an unused output placeholder.
            """
            key, grads_accum, loss_accum = carry
            batch_input, batch_labels = micro_batch
            key, subkey = jax.random.split(key)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(
                params, batch_input, batch_labels, subkey
            )
            grads_accum = jax.tree_util.tree_map(
                _add_trees, grads_accum, grads, is_leaf=lambda x: x is None
            )
            loss_accum = loss_accum + loss
            return (key, grads_accum, loss_accum), None

        grads_init = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x) if x is not None else None,
            params,
            is_leaf=lambda x: x is None,
        )
        loss_init = jnp.array(0.0, dtype=jnp.float32)
        (new_key, grads_accum, loss_accum), _ = jax.lax.scan(
            micro_step,
            (state.key, grads_init, loss_init),
            (input_ids, labels),
        )

        scale = 1.0 / labels.size
        loss = loss_accum * scale
        grads = jax.tree_util.tree_map(
            lambda value: _scale_tree(value, scale),
            grads_accum,
            is_leaf=lambda x: x is None,
        )

        grad_norm = optax.tree.norm(grads)

        updates, new_opt_state = optimizer.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_model = eqx.combine(new_params, static)

        new_state = TrainState(
            step=state.step + 1,
            model=new_model,
            opt_state=new_opt_state,
            key=new_key,
        )

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
        }

        return new_state, metrics

    return eqx.filter_jit(train_step) if use_jit else train_step


def make_eval_step(cfg: dict[str, Any]) -> EvalStep:
    """Create the optionally JIT-compiled evaluation step.

    Args:
        cfg: Validated experiment configuration.

    Returns:
        Function that computes mean loss for one validation batch.
    """
    use_jit = cfg.get("jit", True)

    def eval_step(
        model: eqx.Module,
        input_ids: Int[Array, "batch seq"],
        labels: Int[Array, "batch seq"],
    ) -> Float[Array, ""]:
        """Evaluate one batch without stochastic model operations.

        Args:
            model: Model to evaluate.
            input_ids: Input token IDs.
            labels: Target token IDs.

        Returns:
            Scalar mean cross-entropy.
        """
        logits, _ = forward_model(model, input_ids, deterministic=True)
        return cross_entropy_loss(logits, labels)

    return eqx.filter_jit(eval_step) if use_jit else eval_step


def run_validation(
    model: eqx.Module,
    eval_step: EvalStep,
    validation_batches: tuple[
        Int[Array, "val_batches batch seq"],
        Int[Array, "val_batches batch seq"],
    ],
) -> Float[Array, ""]:
    """Evaluate the fixed validation windows and return their token mean.

    Args:
        model: Model to evaluate.
        eval_step: Compiled or eager single-batch evaluation function.
        validation_batches: Fixed input and target validation batches.

    Returns:
        Mean loss across the fixed validation batches.
    """
    input_batches, label_batches = validation_batches
    total_loss = jnp.array(0.0, dtype=jnp.float32)

    for input_ids, labels in zip(input_batches, label_batches, strict=True):
        loss = eval_step(model, input_ids, labels)
        total_loss = total_loss + loss

    return total_loss / input_batches.shape[0]


# =============================================================================
# Model serialization
# =============================================================================


def save_model_artifact(
    run_dir: str | Path,
    model: eqx.Module,
    cfg: dict[str, Any],
) -> Path:
    """Save a model and the configuration needed to load it.

    Args:
        run_dir: Destination run directory.
        model: Trained Megalodon or Llama model.
        cfg: Resolved experiment configuration.

    Returns:
        Path to the serialized model payload.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as file:
        yaml.safe_dump(cfg, file, default_flow_style=False, sort_keys=False)

    if isinstance(model, LlamaLM):
        model_path = run_dir / "model.eqx"
        eqx.tree_serialise_leaves(model_path, model)
    else:
        from megalodon_jax import save_checkpoint as save_megalodon_checkpoint

        model_path = run_dir / "model.safetensors"
        save_megalodon_checkpoint(model, model_path)

    return model_path


def load_model_artifact(
    run_dir: str | Path,
    key: jax.Array,
) -> tuple[eqx.Module, dict[str, Any]]:
    """Load a saved model and its configuration.

    Args:
        run_dir: Run directory containing the model and config files.
        key: PRNG key used to construct or restore the model.

    Returns:
        Loaded model and validated configuration.
    """
    run_dir = Path(run_dir)
    cfg = validate_config(load_config(run_dir / "config.yaml"))
    model_path = run_dir / ("model.safetensors" if cfg["model"] == "megalodon" else "model.eqx")

    if cfg["model"] == "megalodon":
        from megalodon_jax import load_checkpoint as load_megalodon_checkpoint

        model = load_megalodon_checkpoint(model_path, key=key)
    else:
        skeleton = build_model(cfg, key)
        model = eqx.tree_deserialise_leaves(model_path, skeleton)
    return model, cfg
