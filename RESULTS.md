# 1500-Step BF16 Autocast Results

## Setup

- **Dataset**: enwik8 (character-level, ~95M bytes used)
- **Sequence length**: 512 (chunk size 256 for Megalodon)
- **Training steps**: 1500
- **Effective batch size**: 16 (batch_size=1, grad_accum=16)
- **Validation cadence**: every 150 steps, `val_batches=100`
- **Precision**: bf16 autocast (megalodon-jax v0.1.1 precision policy; Llama AMP-style compute)
- **Hardware**: NVIDIA GeForce RTX 5090
- **XLA_FLAGS**: `--xla_gpu_enable_triton_gemm=false`

## Results (bf16 autocast)

| Model         | Parameters | Val Loss @ 1500 | BPC  | Time  |
| ------------- | ---------- | --------------- | ---- | ----- |
| **Megalodon** | 11.28M     | 1.43            | 2.07 | ~3.9m |
| Llama         | 12.49M     | 1.48            | 2.13 | ~3.5m |

## Validation Curve (1500-step run)

| Step | Megalodon | Llama |
| ---- | --------- | ----- |
| 0    | 17.833    | 5.679 |
| 150  | 1.961     | 2.295 |
| 300  | 1.729     | 1.899 |
| 450  | 1.621     | 1.726 |
| 600  | 1.605     | 1.697 |
| 750  | 1.544     | 1.615 |
| 900  | 1.509     | 1.555 |
| 1050 | 1.537     | 1.597 |
| 1200 | 1.484     | 1.533 |
| 1350 | 1.439     | 1.485 |
| 1500 | 1.433     | 1.477 |

## Reproduction

```bash
# Megalodon
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \\
  python train.py --config configs/megalodon_multichunk_512.yaml

# Llama
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \\
  python train.py --config configs/llama_512.yaml
```
