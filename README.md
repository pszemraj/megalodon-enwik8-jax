# megalodon-enwik8-jax

JAX/Equinox port of **[MEGALODON](https://arxiv.org/abs/2404.08801)** character-level language modeling on enwik8. Built on [megalodon-jax](https://github.com/pszemraj/megalodon-jax).

## Results

| Model         | Parameters | Val Loss @ 1500 | BPC  | Time  |
| ------------- | ---------- | --------------- | ---- | ----- |
| **Megalodon** | 11.28M     | **1.43**        | 2.07 | ~3.9m |
| Llama         | 12.49M     | 1.48            | 2.13 | ~3.5m |

See [RESULTS.md](RESULTS.md) for JAX vs PyTorch comparison details.

Stability run details: 1500 steps, validation every 150 steps with `val_batches=100` (defaults in configs).

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
```

## Inference

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python inference.py --ckpt runs/megalodon/checkpoint_final.eqx --prompt "The "
```

## Project Structure

```
megalodon-enwik8-jax/
├── src/megalodon_enwik8_jax/
│   ├── models/
│   │   ├── megalodon.py  # MegalodonLM (wraps megalodon-jax)
│   │   └── llama.py      # Llama baseline
│   └── utils.py          # Data, sampling, training, checkpoints, config
├── configs/              # Training configs
├── data/                 # enwik8.gz (tracked in this repo)
├── vendor/               # PyTorch reference (submodule)
├── train.py              # Training script
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
