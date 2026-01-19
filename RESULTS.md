# Experimental Results (JAX/Equinox)

> [!IMPORTANT]
> **Disclaimer**: This is a sanity check for JAX/PyTorch numerical parity, not a rigorous benchmark. Both models use identical hyperparameters with no per-architecture tuning.

## Setup

- **Dataset**: enwik8 (character-level, ~95M bytes used)
- **Sequence length**: 512 (chunk size 256 for Megalodon)
- **Training steps**: 1500 (defaults in configs)
- **Effective batch size**: 16 (batch_size=1, grad_accum=16; now true accumulation via scan)
- **Validation cadence**: every 150 steps, `val_batches=100`
- **Learning rate**: 4e-4
- **Precision**: bfloat16 compute with fp32 master weights (Megalodon precision policy in megalodon-jax v0.1.1; Llama uses AMP-style bf16 compute), fp32 comparison below
- **Hardware**: NVIDIA GeForce RTX 5090
- **XLA_FLAGS**: `--xla_gpu_enable_triton_gemm=false`

## Results

### JAX (this repo, bf16 baseline)

| Model         | Parameters | Val Loss @ 1500 | BPC  | Time  |
| ------------- | ---------- | --------------- | ---- | ----- |
| **Megalodon** | 11.28M     | 1.43            | 2.07 | ~3.9m |
| Llama         | 12.49M     | 1.48            | 2.13 | ~3.5m |

### PyTorch Reference

| Model         | Parameters | Val Loss @ 1100 | BPC  | Time |
| ------------- | ---------- | --------------- | ---- | ---- |
| **Megalodon** | 11.3M      | **1.45**        | 2.09 | 8m   |
| Llama         | 12.5M      | 1.54            | 2.22 | 3m   |

### JAX vs PyTorch

| Architecture | JAX Loss @ 1500 | PyTorch Loss @ 1100 | Δ Loss | Δ %   |
| ------------ | --------------- | ------------------- | ------ | ----- |
| Megalodon    | 1.43            | 1.45                | -0.02  | -1.2% |
| Llama        | 1.48            | 1.54                | -0.06  | -4.2% |

These are not step-matched; JAX values are from the 1500-step stability run, while PyTorch is the
1100-step reference. With larger validation batches, both JAX models keep drifting down through
1500 with mild oscillation and no sustained late-training blow-up.

## Training Curves (JAX, 1500-step stability run)

| Step | JAX Megalodon | JAX Llama |
| ---- | ------------- | --------- |
| 0    | 17.833*       | 5.679     |
| 150  | 1.961         | 2.295     |
| 300  | 1.729         | 1.899     |
| 450  | 1.621         | 1.726     |
| 600  | 1.605         | 1.697     |
| 750  | 1.544         | 1.615     |
| 900  | 1.509         | 1.555     |
| 1050 | 1.537         | 1.597     |
| 1200 | 1.484         | 1.533     |
| 1350 | 1.439         | 1.485     |
| 1500 | 1.433         | 1.477     |

*High initial loss due to megalodon-jax He initialization vs PyTorch scaled init.

## Precision Comparison (JAX)

Note: these precision comparisons were run prior to the megalodon-jax v0.1.1 migration and
should be refreshed for the new precision policy.

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

- **Megalodon vs Llama**: In the 1500-step stability run, Megalodon is lower at the end.
- **JAX is faster**: ~3.9m vs ~8m for Megalodon at 1500 steps (JAX benefits from XLA fusion)
- **Generation**: Uses megalodon-jax library's `jax.lax.scan`-based generate() for efficient autoregressive decoding
- **Late-training variance**: With `val_batches=100`, both JAX models continue trending down through 1500 with mild oscillation around ~1.5–1.6

## Reproduction

```bash
# Megalodon
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \\
  python train.py --config configs/megalodon_multichunk_512.yaml

# Llama
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \\
  python train.py --config configs/llama_512.yaml
```

The 1500-step stability run uses the default configs as-is.
