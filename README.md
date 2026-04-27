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
- prompt-only planning: learned text-to-plan module trained before the main video experiment

## Included components

- manifest-based dataset loading for existing videos
- optional sidecar loading for tracks / depth / visibility arrays
- unified scene-hint builder
- learned text-to-plan planner
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
  planner.py
  planner_learned.py
  trainer.py
  trainer_cogvideox_v2.py
  trainer_planner.py
  inference.py
  inference_with_planner.py
  text.py
  utils.py
configs/
  planner_synthetic.yaml
  planner_real_template.yaml
  lite3d_curriculum.yaml
  cogvideox_2b.yaml
  cogvideox_5b.yaml
scripts/
  build_manifest.py
  train.py
  train_planner.py
  sample.py
  sample_with_planner.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The editable install avoids the earlier `ModuleNotFoundError: anti_chimera` issue.

## Standard experiment order

The repository is now designed around a **planner-first** workflow.

### Stage 1. Train the learned planner first

Synthetic planner training:

```bash
python scripts/train_planner.py --config configs/planner_synthetic.yaml
```

Real-data planner training template:

```bash
python scripts/train_planner.py --config configs/planner_real_template.yaml
```

This produces a planner checkpoint such as:

```text
outputs/planner_synthetic/checkpoints/last.pt
```

or

```text
outputs/planner_real/checkpoints/last.pt
```

### Stage 2. Run the main video experiment

Lite synthetic curriculum experiment:

```bash
python scripts/train.py --config configs/lite3d_curriculum.yaml
```

CogVideoX 2B experiment:

```bash
python scripts/train.py --config configs/cogvideox_2b.yaml
```

CogVideoX 5B experiment:

```bash
python scripts/train.py --config configs/cogvideox_5b.yaml
```

## Important planner note

The main experiment configs now already contain a `planner.checkpoint` field.

That means validation prompt-only generation will:

1. use the learned planner checkpoint first, if it exists,
2. otherwise fall back to the older heuristic prompt planner.

So the intended execution order is:

```text
train planner -> produce planner checkpoint -> run main video experiment
```

## Build manifests

CSV format:

```text
video,caption
clips/0001.mp4,"a white cat and a black cat fighting"
clips/0002.mp4,"two people crossing each other"
```

### Single manifest

```bash
python scripts/build_manifest.py \
  --video-root /path/to/videos \
  --captions /path/to/captions.csv \
  --out data/train.jsonl
```

### Train / validation split in one command

```bash
python scripts/build_manifest.py \
  --video-root /path/to/videos \
  --captions /path/to/captions.csv \
  --train-out data/train.jsonl \
  --val-out data/val.jsonl \
  --val-ratio 0.1 \
  --seed 42
```

Optional sidecars under `--sidecar-root`:
- `*_tracks.npy`
- `*_depth.npy`
- `*_visibility.npy`
- `*_masks.npy`
- `*_flow.npy`
- `*_occlusion.npy`

## Planner-driven sampling

After both a video model checkpoint and a planner checkpoint exist, planner-driven prompt-only sampling can be run with:

```bash
python scripts/sample_with_planner.py \
  --config configs/lite3d_curriculum.yaml \
  --checkpoint outputs/lite3d_curriculum/checkpoints/last.pt \
  --planner-checkpoint outputs/planner_synthetic/checkpoints/last.pt \
  --prompt "a white cat and a black cat colliding" \
  --out outputs/samples/planned_sample.gif
```

## Important note

The code is now organized so that the custom research logic is split into two explicit pieces:

1. **learned text-to-plan**
2. **anti-chimera scene-conditioned video generation**

In other words, the intended full system is:

```text
prompt -> learned planner -> scene hint -> video diffusion model
```

The backbone, tokenizer, VAE, and scheduler are still intended to be reused from CogVideoX whenever the heavyweight path is used.
