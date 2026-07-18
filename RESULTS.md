# Paper-scaled Megalodon versus Llama 2 on enwik8

The comparison supports the qualitative paper claim at this small scale: Megalodon reaches lower held-out loss at equal training tokens, including after a second independently declared horizon. It does not win on throughput. On this 512-token, roughly 11M-parameter, single-GPU workload, Llama is about 1.6× faster.

The recorded runs used `megalodon-jax==0.2.0`; the repository supports the 0.2 API.

## Architecture and comparison basis

The primary configs use the small model scale `d=384`, `L=6`, then preserve the released/paper geometry where it scales cleanly:

| Field | Megalodon | Llama 2 baseline | Scaling rationale |
| --- | ---: | ---: | --- |
| Model width / layers | 384 / 6 | 384 / 6 | Same width and depth, matching the paper's comparison basis |
| Attention heads | 1 | 3 | Megalodon uses a minimal reduced head count; Llama keeps 128-wide heads |
| Q/K projection width | 96 | 384 | Megalodon keeps `z=d/4`; Llama heads span `d` |
| Value width | 768 | 384 | Megalodon keeps `v=2d` |
| SwiGLU hidden width | 768 | 1,024 | Megalodon keeps `2d`; Llama uses rounded `8d/3` |
| CEMA order | 16 | — | Released Megalodon policy |
| Local attention chunk | 512 | full 512 | Same visible context for this experiment |
| RoPE base | 100,000 | 10,000 | Megalodon-7B and Llama 2 policies |
| Parameters | 12,098,112 | 10,818,432 | Megalodon is 11.83% larger |

The parameter difference is deliberate. The paper matches width and depth rather than forcing exact parameter equality; its disclosed 7B configurations similarly make Megalodon roughly 9.6% larger by the corresponding formulas.

The Llama baseline uses bias-free projections, 128-wide heads, rounded `8d/3` SwiGLU width, RMSNorm, RoPE base 10,000, untied output weights, and Gaussian `std=0.02` initialization.

## Optimization and LR pilot

Both models use AdamW with `beta1=0.9`, `beta2=0.95`, `eps=1e-8`, weight decay 0.1 on every trainable parameter, global gradient clipping at 1.0, no dropout, and a linear-warmup/cosine-decay schedule. Warmup is 0.5% of the declared horizon, matching the paper's 2,500 warmup updates over approximately 500,000 updates.

Peak learning rate is model-specific. A full-horizon seed-101 pilot used the same batch, data, validation, optimizer, and 1,200-update budget for every candidate. The selection rule was fixed: choose the lowest final validation loss; if the best rate is the upper boundary, expand upward until a worse point brackets the optimum. Seed 101 is not one of the three primary seeds.

| Peak LR | Megalodon loss | Llama loss |
| ---: | ---: | ---: |
| 0.00025 | 1.28935 | 1.29466 |
| 0.00035 | 1.24456 | 1.26171 |
| 0.00050 | 1.20668 | 1.24361 |
| 0.00070 | 1.17204 | **1.22959** |
| 0.00100 | 1.13788 | 1.23355 |
| 0.00140 | 1.12090 | — |
| 0.00200 | **1.11918** | — |
| 0.00280 | 1.15284 | — |

The small Megalodon is substantially more learning-rate-sensitive than Llama. The selected Megalodon rate lies on a broad minimum (`1.12090`, `1.11918`, `1.15284` around the bracket), not an isolated lucky point.

This remains a validation comparison, not a test-set claim. The LR pilot and reported runs use the same fixed validation windows, so the primary seeds measure initialization/training variation after validation-based hyperparameter selection; they do not constitute an untouched final test evaluation.

## Primary equal-token result: 1,200 updates

Each model processes 65,536 target bytes per update and 78,643,200 target bytes over 1,200 updates. Values are mean ± sample standard deviation across paired seeds `[7, 17, 42]`.

| Model | Validation loss | BPC | Synchronized bytes/s |
| --- | ---: | ---: | ---: |
| **Megalodon** | **1.12782 ± 0.00551** | **1.62711 ± 0.00795** | 622,087 ± 218 |
| Llama | 1.17767 ± 0.04172 | 1.69902 ± 0.06019 | **1,008,265 ± 8,996** |

Megalodon's paired loss delta is `-0.04984 ± 0.03916` and its paired BPC delta is `-0.07191 ± 0.05650`, where negative favors Megalodon. Megalodon is lower in all three pairs:

| Seed | Megalodon | Llama | Megalodon − Llama |
| ---: | ---: | ---: | ---: |
| 7 | 1.13378 | 1.18662 | -0.05284 |
| 17 | 1.12679 | 1.21421 | -0.08742 |
| 42 | 1.12290 | 1.13217 | -0.00927 |

The mean validation trajectory shows that this is not a last-checkpoint reversal:

| Update | Megalodon | Llama |
| ---: | ---: | ---: |
| 0 | 6.04443 | 5.66835 |
| 100 | 1.74463 | 2.15733 |
| 300 | 1.36996 | 1.48702 |
| 600 | 1.23198 | 1.29516 |
| 900 | 1.15182 | 1.20296 |
| 1,200 | 1.12782 | 1.17767 |

## Twice-horizon result: 2,400 updates

The extension trains fresh models for 157,286,400 target bytes with 12 warmup updates and cosine decay across the full 2,400-update horizon. It is not a resume from the short run.

| Model | Validation loss | BPC | Synchronized bytes/s |
| --- | ---: | ---: | ---: |
| **Megalodon** | **0.98278 ± 0.00828** | **1.41785 ± 0.01194** | 619,308 ± 3,415 |
| Llama | 1.04649 ± 0.01186 | 1.50977 ± 0.01711 | **1,009,944 ± 2,145** |

Megalodon's paired loss delta is `-0.06372 ± 0.01862`; every long pair favors Megalodon:

| Seed | Megalodon | Llama | Megalodon − Llama |
| ---: | ---: | ---: | ---: |
| 7 | 0.97395 | 1.05165 | -0.07771 |
| 17 | 0.98402 | 1.05488 | -0.07086 |
| 42 | 0.99032 | 1.03290 | -0.04258 |

The longer mean trajectory is:

| Update | Megalodon | Llama |
| ---: | ---: | ---: |
| 0 | 6.04443 | 5.66835 |
| 100 | 1.63653 | 2.10205 |
| 600 | 1.23381 | 1.29448 |
| 1,200 | 1.07321 | 1.13358 |
| 1,800 | 1.00155 | 1.06397 |
| 2,400 | 0.98278 | 1.04649 |

## Performance interpretation

Llama is 1.62× faster in the 1,200-update run and 1.63× faster in the 2,400-update run. That is a real result for this regime, not evidence that the model math is wrong. The paper's throughput advantage appears at much longer contexts, where chunk-local attention avoids Llama's quadratic full-attention cost; at context 512, tiny depth, and one GPU, Megalodon's CEMA and normalization work is overhead while Llama's dense kernels are exceptionally efficient.

## Reproduction

Run the 1,200-update primary comparison:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python compare.py
```

Run the twice-horizon extension:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python compare.py --megalodon-config configs/megalodon_paper_scaled_512_long.yaml --llama-config configs/llama2_paper_scaled_512_long.yaml --output-dir runs/paper_scaled_enwik8_2400
```

Both runners alternate model order by seed, isolate runs in separate processes, synchronize every timed update, and exclude the first ten updates from throughput summaries. Run outputs and the aggregate JSON remain under the ignored output directory.
