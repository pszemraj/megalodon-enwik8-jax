# 1200-Step BF16 Autocast Results

## Setup

- **Dataset**: enwik8 (character-level, ~95M bytes used)
- **Sequence length**: 512 (chunk size 256 for Megalodon)
- **Training steps**: 1200
- **Effective batch size**: 16 (batch_size=1, grad_accum=16)
- **Validation cadence**: every 100 steps, `val_batches=100`
- **Precision**: bf16 autocast (megalodon-jax v0.1.1 precision policy; Llama AMP-style compute)
- **Hardware**: NVIDIA GeForce RTX 5090
- **XLA_FLAGS**: `--xla_gpu_enable_triton_gemm=false`

## Results (bf16 autocast)

| Model         | Parameters | Val Loss @ 1200 | BPC  | Time  |
| ------------- | ---------- | --------------- | ---- | ----- |
| **Megalodon** | 11.28M     | 1.49            | 2.15 | ~3.2m |
| Llama         | 12.49M     | 1.53            | 2.21 | ~3.1m |

## Validation Curve (1200-step run)

| Step | Megalodon | Llama |
| ---- | --------- | ----- |
| 0    | 17.833    | 5.679 |
| 100  | 2.100     | 2.537 |
| 200  | 1.862     | 2.147 |
| 300  | 1.737     | 1.916 |
| 400  | 1.675     | 1.806 |
| 500  | 1.608     | 1.700 |
| 600  | 1.567     | 1.646 |
| 700  | 1.499     | 1.564 |
| 800  | 1.488     | 1.547 |
| 900  | 1.517     | 1.575 |
| 1000 | 1.444     | 1.503 |
| 1100 | 1.444     | 1.508 |
| 1200 | 1.492     | 1.532 |

## Reproduction

```bash
# Megalodon
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \\
  python train.py --config configs/megalodon_multichunk_512.yaml

# Llama
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \\
  python train.py --config configs/llama_512.yaml
```
