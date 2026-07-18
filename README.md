# megalodon-enwik8-jax

A small JAX/Equinox comparison of [Megalodon](https://arxiv.org/abs/2404.08801) and a paper-scaled Llama 2 baseline on byte-level enwik8. Megalodon is provided by [`megalodon-jax` 0.2](https://github.com/pszemraj/megalodon-jax).

## Results

Three predeclared paired seeds were trained on an RTX 5090 after an equal-budget, independent-seed learning-rate pilot. Values are mean ± sample standard deviation; validation uses the same 100 fixed windows for every run.

| Horizon | Model | Parameters | Validation loss | BPC | Synchronized train bytes/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 1,200 updates | **Megalodon** | 12,098,112 | **1.12782 ± 0.00551** | **1.62711 ± 0.00795** | 622,087 ± 218 |
| 1,200 updates | Llama | 10,818,432 | 1.17767 ± 0.04172 | 1.69902 ± 0.06019 | **1,008,265 ± 8,996** |
| 2,400 updates | **Megalodon** | 12,098,112 | **0.98278 ± 0.00828** | **1.41785 ± 0.01194** | 619,308 ± 3,415 |
| 2,400 updates | Llama | 10,818,432 | 1.04649 ± 0.01186 | 1.50977 ± 0.01711 | **1,009,944 ± 2,145** |

Megalodon has lower loss in all three primary pairs and all three twice-horizon pairs. Llama is 1.62–1.63× faster in synchronized steady-state training at this short context. See [RESULTS.md](RESULTS.md) for the architecture derivation, LR pilot, paired results, validation trajectories, caveats, and interpretation.

## Data

The checked-in `data/enwik8.gz` is the experiment dataset. The loader reads its first 95,000,000 decompressed bytes and preserves the established local split: 85,500,000 training bytes followed by 9,500,000 validation bytes. The paired runner does not download, substitute, or reinterpret the dataset.

## Installation

Python 3.11 or newer is required:

```bash
pip install -e '.[dev]'
```

For NVIDIA GPU installation, install the JAX 0.10-compatible CUDA wheel appropriate for the host before installing this project.

## Run the paired comparison

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python compare.py
```

The default runner uses seeds 7, 17, and 42 in alternating model order. It writes model artifacts, per-run metrics, and `comparison.json` under the ignored `runs/paper_scaled_enwik8_1200/` directory.

If all runs completed but `comparison.json` needs to be regenerated, rerun the same command with `--aggregate-only` to read the existing per-run summaries without launching training again.

The twice-horizon extension uses the same selected peak rates with a 12-update warmup and fresh cosine schedule:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python compare.py --megalodon-config configs/megalodon_paper_scaled_512_long.yaml --llama-config configs/llama2_paper_scaled_512_long.yaml --output-dir runs/paper_scaled_enwik8_2400
```

To run one model, provide a fresh output directory:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py --config configs/megalodon_paper_scaled_512.yaml --run-dir runs/my_megalodon_run
```

Training is intentionally one-shot: there is no `--resume`, optimizer persistence, periodic checkpointing, or in-loop generation. Validation and artifact generation occur outside synchronized train-step timing.

## Inference

Inference loads the config and model from a completed run directory:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python inference.py --run-dir runs/my_megalodon_run --prompt 'The ' --top-k 64 --top-p 0.95
```

Each run has one model payload: native `model.safetensors` for Megalodon or `model.eqx` for Llama. Sampling uses the common `temperature`, `top_k`, and `top_p` controls. Llama inference is an eager qualitative-sampling path and rejects requests whose prompt plus continuation would exceed its precomputed RoPE capacity.

## Experiment setup

- The comparison matches width and depth rather than forcing exact parameters. Megalodon has 12,098,112 parameters versus Llama's 10,818,432, an intentional 11.83% difference that mirrors the paper's comparison basis.
- Megalodon preserves the released/paper `z=d/4`, `v=2d`, `ffn=2d`, CEMA-order-16 geometry. Llama preserves 128-wide heads, rounded `8d/3` SwiGLU width, RoPE base 10,000, and Gaussian `std=0.02` initialization.
- Each update is one physical batch of 128 sequences with no gradient accumulation: 65,536 target bytes per update, 78,643,200 targets in the primary run, and 157,286,400 in the twice-horizon extension.
- Validation remains exactly 100 deterministic windows, evaluated as one independent batch rather than inheriting the training batch shape.
- Parameters, accumulation, softmaxes, loss, and logits follow FP32 policies; ordinary contraction compute uses BF16.
- Both models use AdamW with the paper's betas, epsilon, weight decay, clipping, 0.5% warmup fraction, and cosine decay. Peak rates were selected independently under the same full-horizon pilot budget: `0.002` for Megalodon and `0.0007` for Llama.
- Training samples, compile-only samples, model randomness, and fixed validation windows use independent deterministic streams.
- Steady-state timing blocks on GPU completion, excludes the first 10 updates, and excludes validation and generation.
- The LR pilot and reported runs use the same validation windows, so these are validation results after hyperparameter selection, not untouched test-set claims.

## Project structure

```text
megalodon-enwik8-jax/
├── configs/                       # Model and training configurations
├── data/enwik8.gz                 # Checked-in experiment dataset
├── src/megalodon_enwik8_jax/
│   ├── models/megalodon.py        # megalodon-jax 0.2 adapter
│   ├── models/llama.py            # Paper-scaled Llama 2 baseline
│   └── utils.py                   # Data, training, generation, and model serialization
├── compare.py                     # Paired multi-seed runner and aggregator
├── train.py                       # One-shot training entry point
└── inference.py                   # Artifact loading and generation
```

## Tests

```bash
ruff check .
ruff format --check .
pytest -m 'not gpu'
pytest -m gpu
```

The suite covers paper-scaled parameter counts, Llama initialization, full-versus-cached logits, partial Megalodon chunks, AdamW semantics, schedule endpoints, accumulation equivalence, fixed validation windows, both model serialization round-trips, paired aggregation, and CPU/GPU smoke paths.

## Related projects

- [megalodon-jax](https://github.com/pszemraj/megalodon-jax) — JAX/Equinox Megalodon backend
- [megalodon-enwik8](https://github.com/pszemraj/megalodon-enwik8) — companion PyTorch experiment
- [megalodon-hf](https://github.com/pszemraj/megalodon-hf) — PyTorch/Hugging Face Megalodon

## License

Apache-2.0
