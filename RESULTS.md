# Experimental Results (JAX/Equinox)

> [!IMPORTANT]
> **Disclaimer**: This is a sanity check for JAX/PyTorch numerical parity, not a rigorous benchmark. Both models use identical hyperparameters with no per-architecture tuning.

## Setup

- **Dataset**: enwik8 (character-level, ~95M bytes used)
- **Sequence length**: 512 (chunk size 256 for Megalodon)
- **Training steps**: 1200
- **Effective batch size**: 16 (batch_size=1, grad_accum=16; now true accumulation via scan)
- **Learning rate**: 4e-4
- **Precision**: bfloat16 baseline (Megalodon casts params with fp32-sensitive mask; Llama uses AMP-style bf16 compute), fp32 comparison below
- **Hardware**: NVIDIA GeForce RTX 5090
- **XLA_FLAGS**: `--xla_gpu_enable_triton_gemm=false`

## Results

### JAX (this repo, bf16 baseline)

| Model         | Parameters | Val Loss @ 1100 | BPC  | Time  |
| ------------- | ---------- | --------------- | ---- | ----- |
| **Megalodon** | 11.28M     | 1.62            | 2.34 | ~3.4m |
| Llama         | 12.49M     | 1.61            | 2.32 | ~4.0m |

### PyTorch Reference

| Model         | Parameters | Val Loss @ 1100 | BPC  | Time |
| ------------- | ---------- | --------------- | ---- | ---- |
| **Megalodon** | 11.3M      | **1.45**        | 2.09 | 8m   |
| Llama         | 12.5M      | 1.54            | 2.22 | 3m   |

### JAX vs PyTorch

| Architecture | JAX Loss | PyTorch Loss | Δ Loss | Δ %   |
| ------------ | -------- | ------------ | ------ | ----- |
| Megalodon    | 1.62     | 1.45         | +0.17  | +11.7% |
| Llama        | 1.61     | 1.54         | +0.07  | +4.5% |

JAX Megalodon remains higher loss than PyTorch, while JAX Llama is closer. JAX Megalodon hits a
lower loss mid-training (1.40 @ step 800) before rising again, while PyTorch Megalodon improves
to 1.45 later, suggesting similar modeling capacity with different optimization dynamics.

## Training Curves

| Step | JAX Megalodon | PyTorch Megalodon | JAX Llama | PyTorch Llama |
| ---- | ------------- | ----------------- | --------- | ------------- |
| 0    | 17.92*        | 5.67              | 5.66      | 5.68          |
| 100  | 2.19          | 2.03              | 2.49      | 2.60          |
| 200  | 1.92          | 1.70              | 2.14      | 2.29          |
| 400  | 1.78          | 1.55              | 1.82      | 1.87          |
| 600  | 1.71          | 1.57              | 1.69      | 1.71          |
| 800  | 1.40          | 1.51              | 1.37      | 1.64          |
| 1000 | 1.62          | 1.45              | 1.59      | 1.67          |
| 1100 | 1.62          | 1.45              | 1.61      | 1.54          |

*High initial loss due to megalodon-jax He initialization vs PyTorch scaled init.

## Precision Comparison (JAX)

Llama (bf16 AMP vs fp32):

| Run | 0 | 100 | 200 | 400 | 600 | 800 | 1000 | 1100 |
| --- | --- | --- | --- | --- | --- | --- | ---- | ---- |
| bf16 AMP (torch-init) | 5.6643 | 2.4985 | 2.1428 | 1.8171 | 1.7096 | 1.3758 | 1.6065 | 1.6143 |
| fp32 (torch-init) | 5.6649 | 2.4984 | 2.1431 | 1.8293 | 1.7094 | 1.3747 | 1.6060 | 1.6207 |

Megalodon (bf16 masked vs fp32):

| Run | 0 | 100 | 200 | 400 | 600 | 800 | 1000 | 1100 |
| --- | --- | --- | --- | --- | --- | --- | ---- | ---- |
| bf16 masked params | 17.9235 | 2.1944 | 1.9211 | 1.7774 | 1.7061 | 1.3991 | 1.6226 | 1.6229 |
| fp32 | 17.9270 | 2.1509 | 1.8538 | 1.7313 | 1.6460 | 1.3448 | 1.5696 | 1.5730 |

AMP here means fp32 parameters with bf16 compute; Megalodon keeps CEMA/norms/gamma-beta in fp32.

## Parameter Counts

| Model     | JAX        | PyTorch    | Match |
| --------- | ---------- | ---------- | ----- |
| Megalodon | 11,277,888 | 11,277,696 | ~     |
| Llama     | 12,489,600 | 12,489,792 | ~     |

JAX Llama excludes RoPE frequency buffers from trainable parameter counts and weight decay masks.

## Notes

- **Megalodon vs Llama**: In this run, Llama edges Megalodon by a small margin at step 1100.
- **JAX is faster**: ~3.4m vs ~8m for Megalodon (JAX benefits from XLA fusion)
- **Generation**: Uses megalodon-jax library's `jax.lax.scan`-based generate() for efficient autoregressive decoding
- **Late-training variance**: Both JAX models show increased loss after step 800; likely LR schedule differences

## Reproduction

```bash
# Megalodon
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \\
  python train.py --config configs/megalodon_multichunk_512.yaml

# Llama
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \\
  python train.py --config configs/llama_512.yaml
```
