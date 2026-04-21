# anti_chimera

A runnable research scaffold for chimera-resistant video diffusion using the standard diffusion denoising loss.

## Backbone choice

This repository is now **CogVideoX-first**.

- main backbone family: **CogVideoX**
- supported variants: **2B** and **5B**
- reuse policy: keep pretrained pipeline components and only add the anti-chimera conditioning path

The repository no longer treats UNet as the main path.

## Reuse-first design

This repository prefers reuse over scratch.

- backbone: CogVideoX pipeline components from `diffusers`
- text encoder/tokenizer: reuse the pretrained CogVideoX components
- VAE and scheduler: reuse the pretrained CogVideoX components
- data pipeline: manifest adapter for existing video datasets
- synthetic data: fallback smoke-test path only

## Included components

- manifest-based dataset loading for existing videos
- optional sidecar loading for tracks / depth / visibility arrays
- unified scene-hint builder
- scene-conditioned CogVideoX wrapper
- train / sample scripts
- synthetic fallback dataset

## Layout

```text
anti_chimera/
  data/
    manifest.py
    synthetic_collision.py
    scene_hint.py
  models/
    model.py
  config.py
  trainer.py
  inference.py
  text.py
  utils.py
configs/
  default.yaml
  cogvideox_2b.yaml
  cogvideox_5b.yaml
scripts/
  build_manifest.py
  train.py
  sample.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build a manifest

CSV format:

```text
video,caption
clips/0001.mp4,"a white cat and a black cat fighting"
clips/0002.mp4,"two people crossing each other"
```

Build JSONL manifest:

```bash
python scripts/build_manifest.py \
  --video-root /path/to/videos \
  --captions /path/to/captions.csv \
  --out data/train.jsonl
```

Optional sidecars under `--sidecar-root`:
- `*_tracks.npy`
- `*_depth.npy`
- `*_visibility.npy`

## Train

Choose one of the preset configs and set `model.pretrained_model_name_or_path` to your local path or model id.

```bash
python scripts/train.py --config configs/cogvideox_2b.yaml
```

or

```bash
python scripts/train.py --config configs/cogvideox_5b.yaml
```

If no manifest exists and `data.synthetic_fallback=true`, training falls back to the lightweight synthetic dataset.

## Sample

```bash
python scripts/sample.py \
  --config configs/cogvideox_2b.yaml \
  --checkpoint outputs/cogvideox_2b/checkpoints/last.pt \
  --prompt "a white cat and a black cat colliding" \
  --out outputs/samples/sample.gif
```

## Important note

The code is now designed so that the **only custom research logic** is the anti-chimera scene conditioning. The backbone, text encoder, tokenizer, VAE, and scheduler are all intended to be reused from CogVideoX.
