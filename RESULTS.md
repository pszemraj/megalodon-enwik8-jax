# Experimental Results (JAX/Equinox)

> [!IMPORTANT]
> **Disclaimer**: This is a sanity check for JAX/PyTorch numerical parity, not a rigorous benchmark. Both models use identical hyperparameters with no per-architecture tuning.

## Setup

- **Dataset**: enwik8 (character-level, ~95M bytes used)
- **Sequence length**: 512 (chunk size 256 for Megalodon)
- **Training steps**: 1500 (stability run; base configs default to 1200)
- **Effective batch size**: 16 (batch_size=1, grad_accum=16; now true accumulation via scan)
- **Validation cadence**: every 150 steps, `val_batches=100`
- **Learning rate**: 4e-4
- **Precision**: bfloat16 baseline (Megalodon casts params with fp32-sensitive mask; Llama uses AMP-style bf16 compute), fp32 comparison below
- **Hardware**: NVIDIA GeForce RTX 5090
- **XLA_FLAGS**: `--xla_gpu_enable_triton_gemm=false`

## Results

### JAX (this repo, bf16 baseline)

| Model         | Parameters | Val Loss @ 1500 | BPC  | Time  |
| ------------- | ---------- | --------------- | ---- | ----- |
| **Megalodon** | 11.28M     | 1.47            | 2.12 | ~4.4m |
| Llama         | 12.49M     | 1.48            | 2.13 | ~3.7m |

### PyTorch Reference

| Model         | Parameters | Val Loss @ 1100 | BPC  | Time |
| ------------- | ---------- | --------------- | ---- | ---- |
| **Megalodon** | 11.3M      | **1.45**        | 2.09 | 8m   |
| Llama         | 12.5M      | 1.54            | 2.22 | 3m   |

### JAX vs PyTorch

| Architecture | JAX Loss @ 1500 | PyTorch Loss @ 1100 | Δ Loss | Δ %   |
| ------------ | --------------- | ------------------- | ------ | ----- |
| Megalodon    | 1.47            | 1.45                | +0.02  | +1.1% |
| Llama        | 1.48            | 1.54                | -0.06  | -4.2% |

These are not step-matched; JAX values are from the 1500-step stability run, while PyTorch is the
1100-step reference. With larger validation batches, both JAX models keep drifting down through
1500 with mild oscillation and no sustained late-training blow-up.

## Training Curves (JAX, 1500-step stability run)

| Step | JAX Megalodon | JAX Llama |
| ---- | ------------- | --------- |
| 0    | 17.832*       | 5.679     |
| 150  | 2.015         | 2.295     |
| 300  | 1.794         | 1.903     |
| 450  | 1.680         | 1.727     |
| 600  | 1.661         | 1.696     |
| 750  | 1.600         | 1.616     |
| 900  | 1.546         | 1.555     |
| 1050 | 1.581         | 1.597     |
| 1200 | 1.520         | 1.532     |
| 1350 | 1.473         | 1.481     |
| 1500 | 1.466         | 1.476     |

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

- **Megalodon vs Llama**: In the 1500-step stability run, Megalodon is marginally lower at the end.
- **JAX is faster**: ~4.4m vs ~8m for Megalodon at 1500 steps (JAX benefits from XLA fusion)
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

To reproduce the 1500-step stability run above, set `num_batches: 1500` in the config (or a local copy).
