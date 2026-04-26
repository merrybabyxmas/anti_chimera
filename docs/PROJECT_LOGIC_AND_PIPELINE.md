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

Concretely, the implementation tries to achieve that by:

1. reusing a pretrained CogVideoX backbone,
2. constructing a structured scene hint from tracks/depth/visibility,
3. injecting that scene hint into latent-space video generation.

---

# 2. High-Level Design Principles

The current codebase follows these major design principles.

## 2.1 Reuse over scratch

The repository intentionally avoids rebuilding the entire video generation stack from scratch.

The intended reuse policy is:

- **Backbone**: reuse CogVideoX components from `diffusers`
- **Tokenizer / Text Encoder**: reuse pretrained components from the CogVideoX pipeline
- **VAE**: reuse pretrained latent video autoencoding components
- **Scheduler**: reuse the diffusion scheduler shipped with the CogVideoX pipeline
- **Data ingestion**: use a manifest-based adapter for existing datasets instead of forcing a custom dataset format

The custom logic is meant to be concentrated in only two places:

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

The implementation aims to keep the loss as the ordinary denoising regression objective in latent space:

- encode videos into latents,
- add noise according to the scheduler,
- predict the scheduler target (`epsilon` or velocity depending on scheduler configuration),
- optimize MSE.

No explicit depth loss, no explicit instance loss, and no separate amodal loss are currently implemented in the code.

---

# 3. Repository Structure

Current main structure:

```text
anti_chimera/
  __init__.py
  config.py
  inference.py
  text.py
  trainer.py
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
  default.yaml
  cogvideox_2b.yaml
  cogvideox_5b.yaml
scripts/
  build_manifest.py
  train.py
  sample.py
docs/
  PROJECT_LOGIC_AND_PIPELINE.md
```

The practical logic of the project lives in the following files:

- `anti_chimera/data/manifest.py`
- `anti_chimera/data/scene_hint.py`
- `anti_chimera/data/synthetic_collision.py`
- `anti_chimera/models/model.py`
- `anti_chimera/trainer.py`
- `anti_chimera/inference.py`
- `scripts/build_manifest.py`
- `scripts/train.py`
- `scripts/sample.py`

---

# 4. Data Layer

The repository currently supports two dataset paths:

1. **Manifest-based dataset** (intended main path)
2. **Synthetic collision dataset** (fallback smoke-test path)

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

### Optional arrays

If sidecar files exist:

- tracks are expected as shape `(T, max_objects, 4)`
- depth is expected as shape `(T, H, W)`
- visibility is expected as shape `(T, max_objects)`

If missing, zero tensors of the expected shapes are returned.

### Current strengths

- easy to adapt existing datasets,
- allows gradual addition of side information,
- matches the reuse-first philosophy.

### Current limitations

- no validation of sidecar shape consistency beyond implied shapes,
- no robust corrupt-video handling,
- no caching,
- no manifest schema validation,
- no frame-rate normalization beyond uniform frame subsampling.

## 4.2 SyntheticCollisionDataset

Implemented in `anti_chimera/data/synthetic_collision.py`.

### Purpose

This dataset exists as a fallback path when no manifest is available. It is not intended to be the main experimental dataset, but rather a lightweight self-contained source for smoke testing and minimal development.

### Generated content

It synthesizes simple moving shapes with:

- colors,
- basic shape categories,
- trajectories,
- depth ordering,
- visible overlap,
- collision-like motion templates.

### Output

Each sample returns:

- `video`
- `caption`
- `entities`
- `tracks`
- `depth`
- `instance_map`
- `visibility`

This mirrors the structure expected by the scene hint builder.

### Current role in the project

This dataset is not the intended final training path. It is best understood as:

- a smoke-test dataset,
- a debugging fallback,
- a development scaffold when no real manifest exists.

---

# 5. Manifest Construction Script

Implemented in `scripts/build_manifest.py`.

## Purpose

To convert an existing directory of videos plus a caption table into one or two manifest files.

## Inputs

- `--video-root`
- `--captions`
- optional `--sidecar-root`
- either `--out` or `--train-out`
- optional `--val-out`
- optional `--val-ratio`
- optional `--seed`
- optional `--strict`

## Caption table format

The script supports:

- CSV
- JSONL

At minimum the rows must contain:

- `video`
- `caption`

The `video` field is matched either by:

- relative path from `video_root`, or
- bare filename.

## Optional splitting

The script can:

- write a single manifest, or
- split data into train and validation manifests.

## Optional sidecars

If `--sidecar-root` is provided, it looks for:

- `*_tracks.npy`
- `*_depth.npy`
- `*_visibility.npy`

and writes those paths into the manifest if they exist.

## Importance to the overall project

This script is the bridge between the project code and external datasets. In practice, this is the intended primary way to onboard real data into the repository.

---

# 6. Scene Hint Construction

Implemented in `anti_chimera/data/scene_hint.py`.

This is one of the two core custom parts of the repository.

## 6.1 Why the scene hint exists

The project does not try to solve chimera by supervising many task-specific heads. Instead, it tries to alter the model’s conditioning structure.

The scene hint is the main representation used for that.

It transforms available structural information into a single dense tensor that can be injected into the backbone.

## 6.2 Inputs used by the builder

The builder currently uses:

- `tracks`
- `depth`
- `visibility`

and ignores more advanced information such as instance masks or amodal masks, even if they might be available elsewhere.

## 6.3 Output format

The final output is a tensor with shape:

```text
[C_cond, T, H, W]
```

where the condition channels are formed by concatenating:

1. entity grounding maps,
2. depth-bin occupancy maps,
3. visibility maps.

## 6.4 Current channel breakdown

The current effective number of condition channels is:

```text
max_objects + depth_bins + max_objects
```

That is:

- one grounding plane per object slot,
- one plane per depth bin,
- one visibility plane per object slot.

## 6.5 Entity grounding

The grounding builder rasterizes normalized `(x1, y1, x2, y2)` tracks into coarse rectangular occupancy regions.

This is simple and cheap, but also very coarse.

## 6.6 Depth binning

The depth map is resized and converted into discrete depth bins.

Lower depth means closer in the current convention.

## 6.7 Visibility map

Visibility is simply broadcast per-object over the whole spatial grid for each frame.

This is semantically weak but structurally simple.

## 6.8 Current limitations

This is an important section because the current scene hint is functional but still very primitive.

### Limitations

- no amodal mask support,
- no instance mask support,
- no learned object queries in the builder,
- no geometric relationship encoding beyond simple depth bins,
- grounding is box-shaped rather than contour-aware,
- visibility is spatially broadcast rather than localized.

Despite those limitations, the current builder already enforces the central project constraint:

> every sample, regardless of source, must be converted to a single unified conditioning tensor.

---

# 7. Text and Prompt Parsing

Implemented in `anti_chimera/text.py`.

The file currently contains two different conceptual roles:

1. `PromptParser`
2. `HFTextEncoder`

## 7.1 PromptParser

This is a lightweight rule-based parser used primarily for heuristic entity extraction in null-condition construction and simple prompt decomposition.

It uses:

- regex tokenization,
- a small stopword list,
- a small set of entity head words.

This parser is intentionally lightweight, but it is not robust enough to serve as a serious linguistic parser for complex prompts.

## 7.2 HFTextEncoder

This class loads:

- a Hugging Face tokenizer,
- a Hugging Face text model,

and returns token-level hidden states.

## 7.3 Current relationship to CogVideoX

At the current stage, there is a conceptual tension in the repository:

- the model wrapper reuses the tokenizer and text encoder shipped with the CogVideoX pipeline,
- while `text.py` still contains a separate general Hugging Face text encoder utility.

So the project already leans toward full CogVideoX reuse, but `text.py` still also serves as a utility layer.

---

# 8. Model Layer

Implemented in `anti_chimera/models/model.py`.

This is the second major custom part of the repository.

## 8.1 Intended role

The class `AntiChimeraVideoDiffusion` is meant to act as a wrapper around a pretrained CogVideoX pipeline while adding a structured conditioning path.

The guiding idea is:

- reuse the pretrained CogVideoX components,
- keep the anti-chimera additions narrow,
- inject structural priors into latent-space generation.

## 8.2 Components reused from CogVideoX

The wrapper loads a `CogVideoXPipeline` and extracts:

- `transformer`
- `tokenizer`
- `text_encoder`
- `vae`
- `scheduler`

These become the backbone components of the project.

## 8.3 Components added by this repository

The custom trainable additions are:

- `cond_to_latent`
- `cond_pool`
- `cond_token_proj`
- `input_gate`
- `token_gate`

These are the anti-chimera pathway.

### 8.3.1 cond_to_latent

Projects the dense scene hint into latent-space channel geometry so it can be added to latent hidden states.

### 8.3.2 cond_pool + cond_token_proj

Compress the condition tensor into a small number of token-like vectors that can be concatenated to encoder hidden states.

### 8.3.3 input_gate / token_gate

Control the strength of scene hint injection.

This means the current anti-chimera design is effectively **dual-path conditioning**:

- latent-space additive conditioning,
- token-space conditioning augmentation.

## 8.4 Forward pass

At a high level, the forward pass does:

1. encode prompts with the reused tokenizer and text encoder,
2. build latent-space condition and token-space condition from the scene hint,
3. add the latent condition to the noisy latents,
4. concatenate text tokens and condition tokens,
5. pass the result to the CogVideoX transformer,
6. return the predicted diffusion target.

## 8.5 Latent video encode/decode helpers

The wrapper also defines:

- `encode_video`
- `decode_latents`
- `add_noise`
- `prediction_target`
- `infer_latent_shape`

These are used by the trainer and inference path.

## 8.6 Current strengths

- follows the reuse-first principle,
- keeps the custom logic small,
- makes anti-chimera conditioning explicit,
- isolates the project novelty into a wrapper rather than a full new backbone.

## 8.7 Current weaknesses

This file is also the largest source of practical risk in the repository.

### Major weaknesses

1. **Heavy eager loading**
   The wrapper loads the entire CogVideoX pipeline during initialization. This makes startup very heavy and brittle.

2. **Large memory footprint**
   5B loading is likely to be unstable without more careful offloading, dtype control, and parameter freezing strategy.

3. **Config mismatch risk**
   The class signature accepts subfolder-related arguments, but the current implementation conceptually assumes full pipeline loading and does not currently expose a more surgical loading path.

4. **Shape inference cost**
   `infer_latent_shape()` encodes a dummy video through the VAE, which is expensive for something that is conceptually just a shape query.

5. **Backbone-scale training realism**
   The wrapper can technically be optimized, but the repository does not yet implement the full suite of large-model training safeguards needed for stable 2B/5B fine-tuning.

---

# 9. Training Pipeline

Implemented in `anti_chimera/trainer.py`.

## 9.1 Overall structure

The training pipeline currently does the following:

1. choose device,
2. create output folders,
3. build train/validation datasets,
4. create dataloaders,
5. create the scene hint builder,
6. instantiate the `AntiChimeraVideoDiffusion` model,
7. create optimizer and gradient scaler,
8. iterate through epochs and batches.

## 9.2 Dataset selection logic

The training code first tries to use the manifest dataset.

If:

- `data.type == manifest`, and
- `manifest_path` exists,

then it uses `ManifestVideoDataset`.

Otherwise, if `synthetic_fallback` is enabled, it uses `SyntheticCollisionDataset`.

This means the intended production path is manifest-based, and the synthetic path is only a fallback.

## 9.3 Batch collation

The collate function stacks:

- `video`
- `tracks`
- `depth`
- `visibility`

and keeps captions as a Python list.

## 9.4 Conditioning build step

For each batch item, `build_batch_cond()` calls `SceneHintBuilder.build()` and stacks the result into a batch condition tensor.

This is the main point where all source datasets converge into a single condition representation.

## 9.5 Latent-space training flow

For each batch:

1. normalize videos from `[0,1]` to `[-1,1]`,
2. encode the videos to latents via the VAE,
3. sample a timestep,
4. sample Gaussian noise,
5. add noise through the scheduler,
6. compute the scheduler-specific target,
7. run the anti-chimera model forward,
8. compute MSE,
9. backpropagate,
10. optimizer step.

This is exactly where the “standard diffusion loss only” principle is enforced in code.

## 9.6 Checkpointing

At the end of each epoch, the trainer saves:

- `last.pt`
- `epoch_XXX.pt`

Each checkpoint contains:

- model state dict,
- optimizer state dict,
- config,
- epoch.

## 9.7 Current strengths

- simple and readable,
- preserves the central diffusion training objective,
- minimal branching once data enters the model.

## 9.8 Current weaknesses

This file is currently a conceptual training loop rather than a hardened large-scale CogVideoX fine-tuning system.

### Missing pieces

- no gradient checkpointing,
- no accelerator-based distributed training,
- no bf16 strategy,
- no CPU offload strategy,
- no LoRA / PEFT strategy,
- no explicit restriction that only anti-chimera layers should be trained by default,
- no memory-aware handling for 5B.

In practice, this means the current trainer is a valid prototype but not yet a robust large-model finetuning stack.

---

# 10. Inference Pipeline

Implemented in `anti_chimera/inference.py`.

## 10.1 Null condition generation

Inference currently supports a null-condition path built from the prompt alone.

It does this by:

- parsing entities from the prompt heuristically,
- generating a simple collision-like motion prior,
- constructing default tracks, depth, and visibility,
- converting that into a scene hint.

This is a very lightweight inference-time conditioning path.

## 10.2 Sampling loop

Sampling proceeds as follows:

1. build the null condition,
2. set scheduler timesteps,
3. infer latent shape,
4. initialize Gaussian latent noise,
5. iterate reverse diffusion steps,
6. decode final latents through the VAE,
7. map output back to `[0,1]`.

## 10.3 Current strengths

- simple and easy to inspect,
- enough to demonstrate end-to-end path existence,
- keeps the custom logic lightweight.

## 10.4 Current weaknesses

- guidance scale is not meaningfully realized as a real classifier-free guidance implementation,
- latent shape inference is too expensive,
- the null condition is heuristic and weak,
- not aligned with full CogVideoX generation quality expectations.

---

# 11. Configuration Layer

Implemented through YAML configs in `configs/`.

## 11.1 default.yaml

Current `default.yaml` is effectively CogVideoX-first and includes:

- manifest-based data defaults,
- fallback synthetic settings,
- model backend set to `cogvideox`,
- a placeholder pretrained path.

## 11.2 cogvideox_2b.yaml

Dedicated preset intended for CogVideoX 2B.

## 11.3 cogvideox_5b.yaml

Dedicated preset intended for CogVideoX 5B.

## 11.4 Practical interpretation

Even though the repository supports both variants conceptually, the current practical recommendation is:

- use **2B first**,
- treat **5B as an advanced path**,
- do not assume 5B is currently production-ready in this repository without extra memory and environment work.

---

# 12. Scripts Layer

## 12.1 scripts/train.py

This is the CLI entrypoint for training.

It:

- loads config,
- sets seed,
- calls `train(config)`.

## 12.2 scripts/sample.py

This is the CLI entrypoint for inference.

It:

- loads config,
- builds the model,
- loads checkpoint,
- calls `sample_video`,
- writes a GIF.

## 12.3 scripts/build_manifest.py

This is the CLI entrypoint for onboarding existing data into the project.

It is one of the most practically useful scripts in the repository because it bridges external datasets into the project’s unified format.

---

# 13. End-to-End Execution Flow

This section summarizes the current intended workflow from raw data to generated sample.

## 13.1 Data preparation

1. Prepare a directory of videos.
2. Prepare a caption file (`csv` or `jsonl`).
3. Optionally prepare sidecars for tracks/depth/visibility.
4. Run `scripts/build_manifest.py`.

## 13.2 Training

1. Load config.
2. Build manifest dataset or fallback synthetic dataset.
3. Build scene hint tensors per batch.
4. Normalize videos.
5. Encode videos to latents with CogVideoX VAE.
6. Add scheduler noise.
7. Predict denoising target with CogVideoX transformer + anti-chimera conditioning.
8. Optimize MSE.
9. Save checkpoints.

## 13.3 Inference

1. Load config and checkpoint.
2. Build a null scene hint from the prompt.
3. Sample latent noise.
4. Reverse diffuse through the CogVideoX scheduler and transformer.
5. Decode with the VAE.
6. Save as GIF.

---

# 14. Environment and Systems Realities

This project currently depends on very heavy model infrastructure.

## 14.1 The most important environment reality

CogVideoX 2B/5B is not a lightweight dependency.

The repository may fail even before training if:

- the model download is interrupted,
- the CUDA driver is too old for the installed PyTorch build,
- memory is insufficient,
- the `diffusers` version does not properly support CogVideoX,
- the backbone is loaded too eagerly.

## 14.2 Practical recommendation

At the current stage:

- prefer **Python 3.10 or 3.11**,
- prefer a PyTorch build compatible with the installed NVIDIA driver,
- start with **CogVideoX-2B**,
- do not treat 5B as the first debugging target,
- verify `torch.cuda.is_available()` before attempting model loading.

---

# 15. Mismatch Between Intended Design and Current Code Reality

This section is deliberately explicit.

The repository already captures the intended project direction, but not every layer is equally mature.

## Mature / directionally correct parts

- reuse-first philosophy,
- manifest-based dataset ingestion,
- unified scene hint interface,
- anti-chimera conditioning path concept,
- latent-space diffusion training structure.

## Still provisional or incomplete parts

- large-model memory strategy,
- robust CogVideoX loading strategy,
- subfolder-aware loading semantics,
- practical 5B training stability,
- strong prompt/entity parsing,
- real CFG inference path,
- latent shape inference efficiency,
- richer scene hint representation.

---

# 16. Practical Lite Backend

To make the repository runnable end-to-end on a single GPU without waiting for large CogVideoX downloads, the project now also includes a lightweight `lite3d` backend.

## 16.1 What it changes

- trains directly in pixel space,
- keeps the scene-hint conditioning path,
- uses a small learned prompt encoder instead of a remote text model,
- uses a minimal DDIM-style scheduler,
- supports classifier-free guidance during sampling,
- works well as the first validation target on synthetic collision videos.

## 16.2 Why it matters

This backend does not replace the CogVideoX direction. It gives the repository a practical, fully self-contained path for:

- environment verification,
- quick iteration,
- actual GPU training,
- actual GPU sampling,
- regression testing of the conditioning logic.

In other words, it is the execution path that lets the research scaffold become a real experiment loop before the heavyweight backbone is brought in.

---

# 16. Current Bottlenecks

From a code-review and execution perspective, the main bottlenecks are:

1. **model initialization is too eager and too heavy**,
2. **the trainer is not yet hardened for large CogVideoX finetuning**,
3. **5B is too ambitious as a first execution target**,
4. **the inference path is functional but still heuristic**,
5. **the condition representation is useful but still very coarse**.

---

# 17. What the Repository Currently Is

The best way to characterize the repository today is this:

> It is a CogVideoX-first anti-chimera research prototype that already has the right conceptual modularization, but still needs systems-level stabilization before it can be treated as a production-grade large-model fine-tuning framework.

That means:

- the main logic is present,
- the project direction is coherent,
- the codebase is already meaningful as a research scaffold,
- but the training stack still needs another stabilization pass before large-scale experiments are truly smooth.

---

# 18. Recommended Near-Term Development Priorities

If the project is continued from the current state, the most valuable next steps are:

1. make CogVideoX loading lazier and more controllable,
2. default to 2B and explicitly treat 5B as advanced mode,
3. introduce PEFT / LoRA or another restricted-parameter fine-tuning route,
4. remove dummy VAE shape inference from sampling,
5. strengthen manifest/sidecar validation,
6. improve prompt/entity parsing,
7. make guidance scale meaningful in inference,
8. enrich scene hints with richer geometric or amodal structure.

---

# 19. Final Summary

The current `anti_chimera` repository is organized around a simple but strong idea:

- reuse a pretrained CogVideoX backbone,
- build a structured scene hint,
- inject that hint into latent video generation,
- train with a standard diffusion denoising loss.

The project already has:

- a reusable data ingestion pipeline,
- a unified condition builder,
- a CogVideoX-based wrapper model,
- a latent-space training loop,
- a basic inference path.

Its main remaining challenge is not conceptual clarity, but **systems robustness**: large-model loading, memory management, realistic fine-tuning policy, and practical execution stability.

This document should be treated as the canonical record of the repository’s current logic and execution pipeline at this stage of development.


---

# 10. Current Backbone and Real-Data Training Plan

As of 2026-04-25, the intended real-data path is:

1. use the pretrained CogVideoX-2B pipeline from `diffusers`,
2. keep the backbone frozen by default,
3. train only the anti-chimera conditioning path and lightweight adapters,
4. prepare real data through `scripts/prepare_viddata.py`,
5. run the manifest-based train / sample scripts on `configs/cogvideox_viddata*.yaml`.

This is the main separation of responsibilities:

- **backbone / tokenizer / VAE / scheduler**: reused from the pretrained CogVideoX stack,
- **custom logic**: scene hint building, conditioning injection, data manifesting, and experiment logging,
- **lite3d path**: smoke-test only, kept for fast local validation and debugging.

The first real dataset selected for the project is `Databoost/VidData`, because it already ships MP4 files and caption metadata, so it can be turned into manifests without any extra scraping or format conversion.
