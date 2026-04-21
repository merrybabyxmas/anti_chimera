# anti_chimera

A runnable research scaffold for chimera-resistant video diffusion using the standard diffusion denoising loss.

## Reuse-first design

This repository now prefers **reuse over scratch**.

- backbone: `diffusers.UNet3DConditionModel`
- text encoder: Hugging Face `transformers`
- data pipeline: manifest adapter for existing video datasets
- synthetic data: fallback smoke-test path only

## Included components

- manifest-based dataset loading for existing videos
- optional sidecar loading for tracks / depth / visibility arrays
- unified scene-hint builder
- scene-conditioned wrapper around a diffusers 3D UNet
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

Set `data.manifest_path` in `configs/default.yaml`.

```bash
python scripts/train.py --config configs/default.yaml
```

If no manifest exists and `data.synthetic_fallback=true`, training falls back to the lightweight synthetic dataset.

## Sample

```bash
python scripts/sample.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/last.pt \
  --prompt "a white cat and a black cat colliding" \
  --out outputs/samples/sample.gif
```

## Backbone modes

To reuse a pretrained diffusers UNet, set:

```yaml
model:
  backend: diffusers_unet3d
  pretrained_model_name_or_path: /path/to/model-or-hf-id
  unet_subfolder: unet
  text_encoder_name_or_path: openai/clip-vit-base-patch32
```

If `pretrained_model_name_or_path: null`, the code still uses a diffusers UNet class, but without pretrained weights.
