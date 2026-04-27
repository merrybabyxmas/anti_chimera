# anti_chimera Project Logic and End-to-End Pipeline

This document records the **current project logic, implementation structure, execution flow, data assumptions, training path, inference path, environment constraints, and known limitations** of the `anti_chimera` repository.

The goal of this file is not to present an idealized future design, but to capture the repository **as it currently exists**, including where the code is already aligned with the intended research direction and where the implementation is still provisional.

---

# 1. Project Objective

The repository implements a research scaffold for reducing **chimera artifacts** in multi-entity video generation.

In this project, a chimera artifact refers to a failure mode in which multiple entities that should remain distinct become visually entangled when they approach, overlap, or occlude each other. Typical examples include:

- two entities collapsing into one hybrid entity,
- appearance leakage from one entity into another,
- identity swapping after overlap,
- unstable re-emergence after occlusion.

The repository is built around one central thesis:

> Keep the diffusion objective simple, and reduce chimera artifacts by changing the conditioning pathway rather than by introducing many task-specific losses.

Concretely, the implementation now tries to achieve that by:

1. reusing a pretrained CogVideoX or lightweight video backbone,
2. training a **learned text-to-plan module** that predicts scene structure from text,
3. constructing a structured scene hint from tracks/depth/visibility/masks/flow/occlusion,
4. injecting that scene hint into latent-space or lightweight video generation.

---

# 2. High-Level Design Principles

The current codebase follows these major design principles.

## 2.1 Reuse over scratch

The repository intentionally avoids rebuilding the entire video generation stack from scratch.

The intended reuse policy is:

- **Backbone**: reuse CogVideoX components from `diffusers`
- **Tokenizer / Text Encoder**: reuse pretrained components from the CogVideoX pipeline when using the heavyweight path
- **VAE**: reuse pretrained latent video autoencoding components
- **Scheduler**: reuse the diffusion scheduler shipped with the CogVideoX pipeline
- **Data ingestion**: use a manifest-based adapter for existing datasets instead of forcing a custom dataset format

The custom logic is concentrated in three places:

- the **learned text-to-plan module**,
- the **scene hint builder**, and
- the **anti-chimera conditioning injection pathway**.

## 2.2 Unified condition interface

Regardless of whether the data comes from:

- an existing real video dataset,
- pseudo-annotated videos,
- or the internal synthetic fallback dataset,

all samples are eventually converted into a **unified scene hint tensor**.

This keeps the training objective fixed while allowing structured priors to enter the model through a single interface.

## 2.3 Standard diffusion training objective

The video generator still keeps the loss as the ordinary denoising regression objective in latent space:

- encode videos into latents,
- add noise according to the scheduler,
- predict the scheduler target (`epsilon` or velocity depending on scheduler configuration),
- optimize MSE.

The planner is supervised separately, while the video generator keeps the standard diffusion objective.

---

# 3. Repository Structure

Current main structure:

```text
anti_chimera/
  __init__.py
  config.py
  inference.py
  inference_with_planner.py
  planner.py
  planner_learned.py
  text.py
  trainer.py
  trainer_cogvideox_v2.py
  trainer_planner.py
  utils.py
  data/
    __init__.py
    manifest.py
    scene_hint.py
    synthetic_collision.py
  models/
    __init__.py
    model.py
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
docs/
  PROJECT_LOGIC_AND_PIPELINE.md
```

The practical logic of the project lives in the following files:

- `anti_chimera/data/manifest.py`
- `anti_chimera/data/scene_hint.py`
- `anti_chimera/data/synthetic_collision.py`
- `anti_chimera/planner_learned.py`
- `anti_chimera/models/model.py`
- `anti_chimera/trainer.py`
- `anti_chimera/trainer_cogvideox_v2.py`
- `anti_chimera/trainer_planner.py`
- `anti_chimera/inference.py`
- `anti_chimera/inference_with_planner.py`
- `scripts/build_manifest.py`
- `scripts/train.py`
- `scripts/train_planner.py`
- `scripts/sample.py`
- `scripts/sample_with_planner.py`

---

# 4. Data Layer

The repository currently supports two dataset paths:

1. **Manifest-based dataset** (intended real-data path)
2. **Synthetic collision dataset** (diagnostic and planner pretraining path)

## 4.1 ManifestVideoDataset

Implemented in `anti_chimera/data/manifest.py`.

### Purpose

This dataset is meant to let the project ingest existing videos with captions and optional structured side information.

### Input format

The dataset expects a JSONL manifest where each line contains at least:

```json
{
  "video_path": "/absolute/or/relative/path/to/video.mp4",
  "caption": "a white cat and a black cat fighting"
}
```

Optional sidecars may also be referenced:

```json
{
  "video_path": "...",
  "caption": "...",
  "tracks_path": "..._tracks.npy",
  "depth_path": "..._depth.npy",
  "visibility_path": "..._visibility.npy"
}
```

The current codebase also expects richer sidecars when available, including masks, optical flow, and occlusion arrays.

### Behavior

For each item, the dataset:

1. resolves `video_path`,
2. loads all frames through `imageio`,
3. resizes frames to `image_size x image_size`,
4. samples exactly `num_frames` uniformly across the clip,
5. returns video in tensor shape:

```text
[C, T, H, W]
```

where pixel values are in `[0, 1]` before normalization in the trainer.

## 4.2 SyntheticCollisionDataset

Implemented in `anti_chimera/data/synthetic_collision.py`.

### Purpose

This dataset exists as a fallback and as the main source for **planner pretraining and controlled anti-chimera diagnostics**.

### Generated content

It now synthesizes:

- multi-object motion,
- visible masks,
- amodal masks,
- depth ordering,
- dense optical flow,
- occlusion maps,
- overlap metadata,
- simple collision / crossing / chasing / circling interactions.

This dataset is therefore used for:

- smoke tests,
- controlled diagnostics,
- learned planner supervision,
- quick validation of sidecar conditioning.

---

# 5. Planner-First Execution Order

This section is now the **canonical execution order** for the repository.

## 5.1 Why planner-first is the default workflow

The project now separates the prompt-only T2V problem into two stages:

1. **text → plan**
2. **plan → video**

That means the learned planner should be trained first, and the main video experiment should then consume the resulting planner checkpoint during prompt-only validation.

## 5.2 Standard order

The intended order is:

```text
train learned planner -> produce planner checkpoint -> run main video experiment
```

### Stage A: train the planner

Synthetic planner pretraining:

```bash
python scripts/train_planner.py --config configs/planner_synthetic.yaml
```

Real-data planner training template:

```bash
python scripts/train_planner.py --config configs/planner_real_template.yaml
```

The outputs are expected at paths such as:

```text
outputs/planner_synthetic/checkpoints/last.pt
outputs/planner_real/checkpoints/last.pt
```

### Stage B: run the main experiment

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

### Stage C: planner-driven prompt-only sampling

```bash
python scripts/sample_with_planner.py \
  --config configs/lite3d_curriculum.yaml \
  --checkpoint outputs/lite3d_curriculum/checkpoints/last.pt \
  --planner-checkpoint outputs/planner_synthetic/checkpoints/last.pt \
  --prompt "a white cat and a black cat colliding" \
  --out outputs/samples/planned_sample.gif
```

## 5.3 How validation now behaves

The main lite trainer and the CogVideoX v2 trainer both support two validation modes:

- **oracle-conditioned**: uses batch sidecar supervision directly
- **prompt-only**: prefers the learned planner checkpoint if `planner.checkpoint` exists in the config, otherwise falls back to the heuristic planner path

This means planner checkpoints are now a first-class part of the experiment loop rather than an optional side utility.

---

# 6. Scene Hint Construction

Implemented in `anti_chimera/data/scene_hint.py`.

The scene hint builder now combines:

- box grounding,
- slot masks,
- depth bins,
- visibility maps,
- dense optical flow,
- dense occlusion maps,
- overlap maps,
- per-slot frontness maps.

The final output remains a unified tensor of shape:

```text
[C_cond, T, H, W]
```

This is the representation consumed by both the lightweight video model and the CogVideoX wrapper.

---

# 7. Learned Text-to-Plan Module

Implemented in `anti_chimera/planner_learned.py`.

## 7.1 Role

The learned planner predicts a coarse structured scene plan directly from text.

Its outputs include:

- slot-wise trajectories,
- slot-wise depth,
- visibility logits,
- rendered masks,
- rendered dense depth,
- rendered flow,
- rendered occlusion,
- entity-count logits.

## 7.2 Why it matters

This is what lets the repository move from:

- oracle-conditioned anti-chimera experiments

to

- genuine prompt-only T2V validation with a learned intermediate planner.

## 7.3 Training

Planner training is supervised using synthetic or real sidecar data and is intentionally separate from diffusion training.

That keeps the generator objective simple while still allowing the project to learn a text-to-structure mapping.

---

# 8. Model Layer

Implemented in `anti_chimera/models/model.py`.

The class `AntiChimeraVideoDiffusion` acts as a backend dispatcher:

- `backend=lite3d`
- `backend=cogvideox`

The generator consumes the scene hint built either from real sidecars, synthetic supervision, heuristic planning, or learned planning.

This is the point where **text-to-plan** and **scene-conditioned video diffusion** meet.

---

# 9. Training Pipelines

## 9.1 Planner training

Implemented in `anti_chimera/trainer_planner.py`.

This trainer learns:

- prompt → tracks
- prompt → depth
- prompt → visibility
- prompt → masks
- prompt → occlusion

using direct supervised losses.

## 9.2 Lite video training

Implemented in `anti_chimera/trainer.py`.

This trainer:

- builds scene hints,
- trains the lightweight video diffusion model,
- logs oracle-conditioned metrics,
- logs prompt-only metrics,
- prefers learned planner checkpoints for prompt-only validation when configured.

## 9.3 CogVideoX conditioner-only training

Implemented in `anti_chimera/trainer_cogvideox_v2.py`.

This trainer:

- keeps the heavyweight generator path focused on conditioning,
- uses manifest-based real data,
- logs oracle-conditioned metrics,
- logs prompt-only metrics,
- prefers learned planner checkpoints for prompt-only validation when configured.

---

# 10. Inference Paths

## 10.1 Standard inference

Implemented in `anti_chimera/inference.py`.

If `planner.checkpoint` exists in the config, prompt-only condition generation now loads the learned planner first.
Otherwise it falls back to the heuristic planner.

## 10.2 Explicit planner-driven inference

Implemented in `anti_chimera/inference_with_planner.py` and `scripts/sample_with_planner.py`.

This path is used when the user explicitly wants to combine:

- a video model checkpoint
- a learned planner checkpoint
- a free-form prompt

into planner-driven prompt-only generation.

---

# 11. Configuration Layer

The configuration layer now contains two distinct config families.

## 11.1 Planner configs

- `configs/planner_synthetic.yaml`
- `configs/planner_real_template.yaml`

These are used for planner training.

## 11.2 Main experiment configs

- `configs/lite3d_curriculum.yaml`
- `configs/cogvideox_2b.yaml`
- `configs/cogvideox_5b.yaml`

These now include a `planner.checkpoint` field so that prompt-only validation will automatically use a trained planner when available.

---

# 12. Environment and Systems Realities

The practical recommendations remain:

- prefer Python 3.10 or 3.11,
- use a PyTorch build compatible with the installed NVIDIA driver,
- start with planner pretraining and lite experiments first,
- only move to CogVideoX after the planner and lite stack are working,
- treat 5B as an advanced path.

---

# 13. Final Summary

The repository is no longer just a scene-conditioned video generator scaffold. It is now organized around a full two-stage research pipeline:

```text
prompt -> learned planner -> scene hint -> video diffusion model
```

The canonical order is now explicit:

1. train the planner,
2. produce a planner checkpoint,
3. run the main video experiment,
4. evaluate both oracle-conditioned and learned-planner-driven prompt-only generation.

This document should be treated as the canonical record of that planner-first workflow.
