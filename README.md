# anti_chimera

A runnable research scaffold for chimera-resistant video diffusion using **only the standard diffusion denoising loss**.

This repository provides:
- an on-the-fly synthetic multi-entity collision dataset
- a unified scene-hint builder (entity grounding + depth bins + visibility)
- an object-centric scene injector
- a compact 3D video denoiser
- training and sampling scripts that run end-to-end

## What this is

This is a **minimal executable research codebase** implementing the core idea:

> Keep the diffusion loss simple, and reduce chimera artifacts by changing the conditioning pathway rather than adding extra task-specific losses.

The code is intentionally lightweight so that training and inference can be run immediately on a single GPU or even on CPU at very small settings.

## Project layout

```text
anti_chimera/
  data/
    synthetic_collision.py   # on-the-fly synthetic videos with collisions/occlusions
    scene_hint.py            # unified condition builder
  models/
    modules.py               # shared blocks
    model.py                 # scene-injected video denoiser
  config.py                  # config loading
  trainer.py                 # train loop
  inference.py               # DDPM sampling
  text.py                    # prompt tokenizer/encoder
  utils.py                   # misc utilities
configs/
  default.yaml
scripts/
  train.py
  sample.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py --config configs/default.yaml
```

## Sample

```bash
python scripts/sample.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/last.pt \
  --prompt "a white circle and a black circle colliding" \
  --out outputs/samples/sample.gif
```

## Notes

- The synthetic dataset uses simple colored objects and motion templates so the code is fully self-contained.
- Scene hints are built with a **single unified pipeline** for every sample.
- The training objective is **standard diffusion MSE only**.
- This scaffold is deliberately compact. It is designed to be the shortest path to a working prototype, not a full-scale pretrained T2V system.
