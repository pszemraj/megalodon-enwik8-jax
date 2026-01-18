# megalodon-enwik8-jax

JAX/Equinox port of **[MEGALODON](https://arxiv.org/abs/2404.08801)** character-level language modeling on enwik8. Built on [megalodon-jax](https://github.com/pszemraj/megalodon-jax).

## Results

Similarly-sized architectures trained on enwik8 for 1200 steps:

| Model         | Parameters | Val Loss @ 1100 | BPC  | Time  |
| ------------- | ---------- | --------------- | ---- | ----- |
| **Megalodon** | 11.3M      | **1.57**        | 2.27 | ~3m   |
| Llama         | 12.9M      | 1.66            | 2.39 | ~3.5m |

See [RESULTS.md](RESULTS.md) for full experimental details and JAX vs PyTorch comparison.

## Quick Start

```bash
pip install -e .

# Train Megalodon
python train.py --config configs/megalodon_multichunk_512.yaml

# Train Llama baseline
python train.py --config configs/llama_512.yaml
```

## Inference

```bash
python inference.py --ckpt runs/megalodon/checkpoint_final.eqx --prompt "The "
```

## Structure

```text
megalodon-enwik8-jax/
├── src/megalodon_enwik8_jax/
│   ├── models/
│   │   ├── megalodon.py  # MegalodonLM wrapper (uses megalodon-jax)
│   │   └── llama.py      # Llama baseline
│   ├── training.py       # TrainState, optimizer, train step
│   ├── generate.py       # Text generation
│   ├── data.py           # enwik8 loading
│   ├── checkpoint.py     # Save/load
│   └── config.py         # YAML config handling
├── configs/
│   ├── megalodon_multichunk_512.yaml
│   ├── llama_512.yaml
│   └── test.yaml
├── train.py
├── inference.py
└── RESULTS.md
```

## Related Projects

- [megalodon-jax](https://github.com/pszemraj/megalodon-jax) - JAX/Equinox Megalodon implementation (this repo's modeling backend)
- [megalodon-hf](https://github.com/pszemraj/megalodon-hf) - PyTorch/HuggingFace Megalodon implementation
- [megalodon-enwik8](https://github.com/pszemraj/megalodon-enwik8) - PyTorch version of this benchmark

## Tests

```bash
pytest tests/ -v
```

## License

MIT (this repo) / Apache-2.0 ([megalodon-jax](https://github.com/pszemraj/megalodon-jax))
