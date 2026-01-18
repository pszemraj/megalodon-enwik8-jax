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

from .models import forward_model
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
        "dtype",
        "jit",
        # Llama-specific
        "dim",
        "depth",
        "heads",
        "dim_head",
        "tied_embedding",
        "ffn_dim_multiplier",
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
        "weight_decay",
        "grad_clip_norm",
        # Data
        "data_path",
        "seq_len",
        # Evaluation
        "validate_every",
        "val_batches",
        "generate_every",
        "generate_prompt_len",
        "generate_length",
        "save_every",
        "temperature",
        "min_p",
        "top_k",
        "top_p",
    }
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    return cfg


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate configuration and apply defaults."""
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

    # Validate dtype
    dtype = cfg.get("dtype", "bf16").lower()
    if dtype not in {"bf16", "fp32"}:
        raise ValueError(f"dtype must be 'bf16' or 'fp32' (no fp16), got '{dtype}'")
    cfg["dtype"] = dtype

    # Llama baseline does not implement dropout; require zeroed values.
    if model == "llama":
        for key in ("dropout", "attention_dropout", "hidden_dropout"):
            if cfg.get(key, 0.0) > 0.0:
                raise ValueError(f"{key} must be 0.0 for Llama baseline.")

    # Megalodon-specific: chunk_size must divide seq_len
    if model == "megalodon":
        seq_len = cfg.get("seq_len", 512)
        chunk_size = cfg.get("chunk_size", seq_len)
        if seq_len > chunk_size and seq_len % chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be <= chunk_size ({chunk_size}) "
                "or divisible by it for Megalodon."
            )

    # Apply defaults for optional fields
    defaults = {
        "seed": 42,
        "jit": True,
        "batch_size": 1,
        "grad_accum_every": 1,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "validate_every": 100,
        "val_batches": 10,
        "generate_every": 100,
        "generate_prompt_len": 128,
        "generate_length": 128,
        "save_every": 500,
        "temperature": 1.0,
        "min_p": 0.1,
    }

    for key, default in defaults.items():
        if key not in cfg:
            cfg[key] = default

    return cfg


def resolve_run_dir(cfg: dict[str, Any], override: str | None = None) -> Path:
    """Resolve the run directory from config or override."""
    if override:
        return Path(override)
    return Path(cfg.get("run_dir", "runs/default"))


def get_dtype(cfg: dict[str, Any]) -> DTypeLike:
    """Get JAX dtype from config."""
    dtype_str = cfg.get("dtype", "bf16").lower()
    return jnp.bfloat16 if dtype_str == "bf16" else jnp.float32


# =============================================================================
# Data utilities
# =============================================================================


def load_enwik8(
    path: str | Path,
    bytes_limit: int = 95_000_000,
    train_split: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Load enwik8 data from gzipped file."""
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
    """Sample a random batch of sequences from data."""
    max_start = len(data_u8) - seq_len - 1

    starts = rng.integers(0, max_start, size=(batch_size,))
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
    """Sample a batch for gradient accumulation."""
    max_start = len(data_u8) - seq_len - 1
    total_seqs = batch_size * grad_accum

    starts = rng.integers(0, max_start, size=(total_seqs,))
    sequences = np.stack(
        [data_u8[start : start + seq_len + 1] for start in starts],
        axis=0,
    )

    input_ids = sequences[:, :-1].astype(np.int32)
    labels = sequences[:, 1:].astype(np.int32)

    input_ids = input_ids.reshape(grad_accum, batch_size, seq_len)
    labels = labels.reshape(grad_accum, batch_size, seq_len)

    return jnp.asarray(input_ids), jnp.asarray(labels)


def encode_prompt(text: str) -> jax.Array:
    """Encode text string to token IDs (bytes)."""
    tokens = np.array(list(text.encode("utf-8")), dtype=np.int32)
    return jnp.asarray(tokens[None, :])


def decode_tokens(tokens: jax.Array) -> str:
    """Decode token IDs to text string."""
    tokens = np.asarray(tokens).flatten()
    bytes_array = bytes(max(32, int(t)) if t < 128 else int(t) for t in tokens)
    return bytes_array.decode("utf-8", errors="replace")


# =============================================================================
# Sampling + generation
# =============================================================================


def apply_temperature(logits: jax.Array, temperature: float) -> jax.Array:
    """Apply temperature scaling to logits."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return logits / temperature


def apply_min_p(logits: jax.Array, min_p: float) -> jax.Array:
    """Apply min-p filtering to logits."""
    if min_p <= 0:
        return logits
    probs = jax.nn.softmax(logits, axis=-1)
    max_probs = probs.max(axis=-1, keepdims=True)
    threshold = min_p * max_probs
    return jnp.where(probs < threshold, -jnp.inf, logits)


def sample_token(
    key: jax.Array,
    logits: jax.Array,
    temperature: float = 1.0,
    min_p: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Sample next token with temperature and min-p filtering."""
    key, sample_key = jax.random.split(key)
    logits = apply_temperature(logits, temperature)
    if min_p > 0:
        logits = apply_min_p(logits, min_p)
    tokens = jax.random.categorical(sample_key, logits, axis=-1)
    return key, tokens


def _generate_llama(
    model: LlamaLM,
    prompt_ids: Int[Array, "batch seq"],
    max_new_tokens: int,
    temperature: float,
    min_p: float,
    key: jax.Array,
) -> Int[Array, "batch total_seq"]:
    """Generate for Llama using Python loop."""
    logits, cache = forward_model(model, prompt_ids, return_cache=True, deterministic=True)

    last_logits = logits[:, -1, :]
    key, next_token = sample_token(key, last_logits, temperature, min_p)
    next_token = next_token[:, None]

    generated = [next_token]

    for _ in range(max_new_tokens - 1):
        logits, cache = forward_model(
            model, next_token, cache=cache, return_cache=True, deterministic=True
        )
        last_logits = logits[:, -1, :]
        key, next_token = sample_token(key, last_logits, temperature, min_p)
        next_token = next_token[:, None]
        generated.append(next_token)

    generated = jnp.concatenate(generated, axis=1)
    return jnp.concatenate([prompt_ids, generated], axis=1)


def generate(
    model: eqx.Module,
    prompt_ids: Int[Array, "batch seq"],
    max_new_tokens: int,
    temperature: float = 1.0,
    min_p: float = 0.0,
    key: jax.Array | None = None,
) -> Int[Array, "batch total_seq"]:
    """Generate text autoregressively with cache-based decoding."""
    if key is None:
        key = jax.random.PRNGKey(0)

    if isinstance(model, LlamaLM):
        return _generate_llama(model, prompt_ids, max_new_tokens, temperature, min_p, key)

    from megalodon_jax import generate as megalodon_generate

    top_p = (1.0 - min_p) if min_p > 0 else None

    result, _, _ = megalodon_generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        key=key,
        temperature=temperature,
        top_p=top_p,
    )
    return result


# =============================================================================
# Params utilities
# =============================================================================


def make_trainable_mask(model: eqx.Module) -> Any:
    """Create a pytree mask of trainable parameters."""
    mask = jax.tree_util.tree_map(eqx.is_inexact_array, model)

    if isinstance(model, LlamaLM):
        layer_count = len(model.layers)
        mask = eqx.tree_at(
            lambda m: [layer.attn.cos for layer in m.layers],
            mask,
            replace=[False] * layer_count,
        )
        mask = eqx.tree_at(
            lambda m: [layer.attn.sin for layer in m.layers],
            mask,
            replace=[False] * layer_count,
        )

    return mask


def cast_trainable(
    model: eqx.Module,
    dtype: jnp.dtype,
    trainable_mask: Any | None = None,
) -> eqx.Module:
    """Cast trainable floating-point parameters to the requested dtype."""
    if trainable_mask is None:
        trainable_mask = make_trainable_mask(model)

    params, static = eqx.partition(model, trainable_mask)

    def _cast_leaf(value: Any) -> Any:
        if value is None:
            return None
        if not jnp.issubdtype(value.dtype, jnp.floating):
            return value
        if value.dtype == dtype:
            return value
        return value.astype(dtype)

    params = jax.tree_util.tree_map(_cast_leaf, params, is_leaf=lambda x: x is None)
    return eqx.combine(params, static)


def assert_trainable_dtype(
    model: eqx.Module,
    dtype: jnp.dtype,
    trainable_mask: Any | None = None,
) -> None:
    """Assert all trainable floating-point parameters match the requested dtype."""
    if trainable_mask is None:
        trainable_mask = make_trainable_mask(model)

    params = eqx.filter(model, trainable_mask)
    leaves = [leaf for leaf in jax.tree.leaves(params) if leaf is not None]

    mismatched = [
        leaf.dtype
        for leaf in leaves
        if jnp.issubdtype(leaf.dtype, jnp.floating) and leaf.dtype != dtype
    ]
    if mismatched:
        unique = sorted({str(item) for item in mismatched})
        raise ValueError(f"Trainable params dtype mismatch. Expected {dtype}, found {unique}.")


def count_trainable_params(
    model: eqx.Module,
    trainable_mask: Any | None = None,
) -> int:
    """Count trainable parameters (excluding non-trainable buffers)."""
    if trainable_mask is None:
        trainable_mask = make_trainable_mask(model)

    params = eqx.filter(model, trainable_mask)
    return sum(leaf.size for leaf in jax.tree.leaves(params) if leaf is not None)


def sample_trainable_dtypes(
    model: eqx.Module,
    trainable_mask: Any | None = None,
    max_samples: int = 3,
) -> list[jnp.dtype]:
    """Return a few representative trainable dtypes for logging."""
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
    """Create initial training state."""
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


def build_optimizer(cfg: dict[str, Any]) -> optax.GradientTransformation:
    """Build optax optimizer matching PyTorch Adam semantics."""
    lr = cfg.get("learning_rate", 1e-3)
    weight_decay = cfg.get("weight_decay", 0.0)
    grad_clip_norm = cfg.get("grad_clip_norm", 1.0)

    transforms = []

    if grad_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip_norm))

    if weight_decay > 0:
        transforms.append(optax.add_decayed_weights(weight_decay))

    transforms.append(optax.adam(learning_rate=lr))

    return optax.chain(*transforms)


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute cross-entropy loss for language modeling."""
    logits_f32 = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)

    batch_size, seq_len, _ = logits.shape
    batch_idx = jnp.arange(batch_size)[:, None]
    seq_idx = jnp.arange(seq_len)[None, :]
    target_log_probs = log_probs[batch_idx, seq_idx, labels]

    return -target_log_probs.mean()


def bpc_from_loss(loss: jax.Array) -> jax.Array:
    """Convert cross-entropy loss to bits-per-character."""
    return loss / jnp.log(2.0)


def make_train_step(
    cfg: dict[str, Any],
    optimizer: optax.GradientTransformation,
    trainable_mask: Any,
) -> Callable[
    [TrainState, Int[Array, "accum batch seq"], Int[Array, "accum batch seq"]],
    tuple[TrainState, dict[str, Array]],
]:
    """Create training step function."""
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
        params, static = eqx.partition(state.model, trainable_mask)

        def loss_fn(
            params: eqx.Module,
            batch_input: Int[Array, "batch seq"],
            batch_labels: Int[Array, "batch seq"],
            key: jax.Array,
        ) -> Float[Array, ""]:
            model = eqx.combine(params, static)
            logits, _ = forward_model(model, batch_input, deterministic=deterministic, key=key)
            return cross_entropy_loss(logits, batch_labels)

        def _add_trees(left: Any, right: Any) -> Any:
            if left is None:
                return None
            return left + right

        def _scale_tree(value: Any, scale: float) -> Any:
            if value is None:
                return None
            return value * scale

        def micro_step(
            carry: tuple[jax.Array, Any, jax.Array],
            micro_batch: tuple[Int[Array, "batch seq"], Int[Array, "batch seq"]],
        ) -> tuple[tuple[jax.Array, Any, jax.Array], None]:
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

        scale = 1.0 / input_ids.shape[0]
        loss = loss_accum * scale
        grads = jax.tree_util.tree_map(
            lambda value: _scale_tree(value, scale),
            grads_accum,
            is_leaf=lambda x: x is None,
        )

        grad_norm = optax.global_norm(grads)

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
    """Create evaluation step function."""
    use_jit = cfg.get("jit", True)

    def eval_step(
        model: eqx.Module,
        input_ids: Int[Array, "batch seq"],
        labels: Int[Array, "batch seq"],
    ) -> Float[Array, ""]:
        logits, _ = forward_model(model, input_ids, deterministic=True)
        return cross_entropy_loss(logits, labels)

    return eqx.filter_jit(eval_step) if use_jit else eval_step


def run_validation(
    model: eqx.Module,
    eval_step: EvalStep,
    val_data: np.ndarray,
    rng: np.random.Generator,
    cfg: dict[str, Any],
) -> Float[Array, ""]:
    """Run validation over multiple batches."""
    val_batches = cfg.get("val_batches", 10)
    batch_size = cfg.get("batch_size", 1)
    seq_len = cfg.get("seq_len", 512)

    total_loss = jnp.array(0.0, dtype=jnp.float32)

    for _ in range(val_batches):
        input_ids, labels = sample_batch(rng, val_data, batch_size, seq_len)
        loss = eval_step(model, input_ids, labels)
        total_loss = total_loss + loss

    return total_loss / val_batches


# =============================================================================
# Checkpoints
# =============================================================================


def save_checkpoint(
    run_dir: str | Path,
    state: TrainState,
    cfg: dict[str, Any],
    tag: str | None = None,
) -> str:
    """Save checkpoint to disk."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    step = int(state.step)
    if tag:
        ckpt_name = f"checkpoint_{tag}.eqx"
    else:
        ckpt_name = f"checkpoint_{step}.eqx"

    ckpt_path = run_dir / ckpt_name

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    checkpoint_data = {
        "model": state.model,
        "opt_state": state.opt_state,
        "step": state.step,
        "key": state.key,
    }

    eqx.tree_serialise_leaves(ckpt_path, checkpoint_data)

    return str(ckpt_path)


def load_checkpoint(
    ckpt_path: str | Path,
    cfg: dict[str, Any],
    key: jax.Array,
    model_builder: callable,
    optimizer: Any,
) -> TrainState:
    """Load checkpoint from disk."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dummy_model = model_builder(cfg, key)
    dummy_state = create_train_state(dummy_model, optimizer, key, step=0)

    skeleton = {
        "model": dummy_state.model,
        "opt_state": dummy_state.opt_state,
        "step": dummy_state.step,
        "key": dummy_state.key,
    }

    loaded = eqx.tree_deserialise_leaves(ckpt_path, skeleton)

    return TrainState(
        step=loaded["step"],
        model=loaded["model"],
        opt_state=loaded["opt_state"],
        key=loaded["key"],
    )


def load_config_from_checkpoint(ckpt_path: str | Path) -> dict[str, Any]:
    """Load config from checkpoint directory."""
    ckpt_path = Path(ckpt_path)

    if ckpt_path.is_file():
        config_path = ckpt_path.parent / "config.yaml"
    else:
        config_path = ckpt_path / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in a run directory."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None

    checkpoints = list(run_dir.glob("checkpoint_*.eqx"))
    if not checkpoints:
        return None

    def get_step(path: Path) -> int:
        name = path.stem
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return -1

    checkpoints.sort(key=get_step)
    return checkpoints[-1]
