# megalodon-enwik8-jax

JAX port of MEGALODON character-level language modeling, comparing against standard Transformers on enwik8.

## Installation

```bash
pip install -e .
```

## Training

```bash
# Quick smoke test
python train.py --config configs/test.yaml

# Full Llama training
python train.py --config configs/llama_512.yaml

# Full Megalodon training
python train.py --config configs/megalodon_multichunk_512.yaml
```

## Inference

```bash
python inference.py --ckpt runs/llama/checkpoint_final.eqx --prompt "The "
```

## Tests

```bash
pytest tests/ -v
```
