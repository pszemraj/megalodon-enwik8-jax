# megalodon-enwik8-jax

JAX/Equinox port of **[MEGALODON](https://arxiv.org/abs/2404.08801)** character-level language modeling on enwik8. Built on [megalodon-jax](https://github.com/pszemraj/megalodon-jax).

## Results

| Model         | Parameters | Val Loss @ 1100 | BPC  | Time  |
| ------------- | ---------- | --------------- | ---- | ----- |
| **Megalodon** | 11.3M      | **1.57**        | 2.27 | ~3m   |
| Llama         | 12.9M      | 1.66            | 2.39 | ~3.5m |

These numbers were produced before the dtype and accumulation fixes below. Re-run after updates
for fresh comparisons. See [RESULTS.md](RESULTS.md) for JAX vs PyTorch comparison details.

## Installation

```bash
git clone https://github.com/pszemraj/megalodon-enwik8-jax.git
cd megalodon-enwik8-jax

# Dataset is already included in data/enwik8.gz

# Install JAX with GPU support (adjust for your CUDA version)
# See: https://jax.readthedocs.io/en/latest/installation.html
pip install -U "jax[cuda12]"

# Install this package
pip install -e .
```

> [!IMPORTANT]
> Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid GPU memory conflicts. This prevents JAX from pre-allocating all GPU memory at startup.

> [!IMPORTANT]
> On some newer GPUs (e.g., RTX 5090) with `jax/jaxlib 0.8.2`, Triton GEMM can fail even for tiny matmuls with `CUDA_ERROR_OUT_OF_MEMORY`.
> If that happens, run with `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` to force cuBLAS.

## Training

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py --config configs/megalodon_multichunk_512.yaml
# or via entrypoint
XLA_PYTHON_CLIENT_PREALLOCATE=false train-megalodon --config configs/megalodon_multichunk_512.yaml
```

## Inference

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python inference.py --ckpt runs/megalodon/checkpoint_final.eqx --prompt "The "
# or via entrypoint
XLA_PYTHON_CLIENT_PREALLOCATE=false infer-megalodon --ckpt runs/megalodon/checkpoint_final.eqx --prompt "The "
```

## Project Structure

```
megalodon-enwik8-jax/
├── src/megalodon_enwik8_jax/
│   ├── models/
│   │   ├── megalodon.py  # MegalodonLM (wraps megalodon-jax)
│   │   └── llama.py      # Llama baseline
│   ├── training.py       # TrainState, optimizer, train step
│   ├── generate.py       # Text generation
│   ├── data.py           # enwik8 loading
│   ├── checkpoint.py     # Save/load
│   └── config.py         # YAML config
├── configs/              # Training configs
├── data/                 # enwik8.gz (tracked in this repo)
├── vendor/               # PyTorch reference (submodule)
├── train.py              # Thin wrapper for package CLI
└── inference.py
```

## Related Projects

- [megalodon-jax](https://github.com/pszemraj/megalodon-jax) - JAX/Equinox Megalodon (modeling backend)
- [megalodon-enwik8](https://github.com/pszemraj/megalodon-enwik8) - PyTorch version of this benchmark
- [megalodon-hf](https://github.com/pszemraj/megalodon-hf) - PyTorch/HuggingFace Megalodon

## Tests

```bash
pytest tests/ -v
```

## License

Apache-2.0
