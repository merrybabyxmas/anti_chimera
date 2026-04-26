# Experiment Log

## 2026-04-25

### Code changes in progress

- Added a lightweight `lite3d` backend so the repository can train and sample without downloading CogVideoX.
- Added a self-contained prompt encoder based on stable hashing for the lite backend.
- Added a minimal DDIM-style scheduler for pixel-space video diffusion.
- Refactored the model wrapper into a backend dispatcher so CogVideoX remains available.
- Updated the scene-hint builder to keep tensors on the active device.
- Hardened the video manifest loader to close readers and handle grayscale frames.
- Upgraded training to log metrics, validate periodically, and save generated GIF samples.
- Added classifier-free guidance during sampling by pairing conditional and unconditional passes.
- Added `configs/lite3d.yaml` for the first real GPU-friendly training run.
- Added `requirements-lite.txt` so the lite path can be installed without heavy CogVideoX dependencies.
- Added resume support to the training CLI.

### Dw39 follow-up

#### Hypothesis

- The pipeline was training, but the sampled videos were still close to noise because the reverse diffusion path was not fully aligned with the scheduled timesteps and the preview path was not easy to inspect.
- The fastest useful improvement was to make the inference outputs visible as PNGs and then verify the final sampling step on the remote GPU run.

#### Code changes made

- Fixed `anti_chimera/diffusion.py` so `SimpleDDPMScheduler.step()` uses the correct previous alpha value for arbitrary inference timesteps.
- Added x0 clamping in `SimpleDDPMScheduler.step()` to stabilize the reverse chain when sampling from full noise.
- Extended `anti_chimera/utils.py` with:
  - MPS / CUDA / CPU aware `default_device()`
  - `save_video_png()` contact-sheet previews for video tensors
- Updated `scripts/sample.py` to save both GIF and PNG previews.
- Added `scripts/refine.py` for reproducible low-timestep denoising / reconstruction runs on the synthetic validation set.
- Updated `anti_chimera/trainer.py` to save PNG previews alongside train/val GIFs and to record those artifact paths in `run_log.md`.
- Added EMA tracking to `trainer.py` and made both training previews and standalone sampling prefer `ema_model` when present.
- Added `configs/lite3d_dw39.yaml` for the 39번 서버 experiment.

#### Remote experiment results on dw39

- Smoke run:
  - output dir: `/home/dongwoo39/projects/anti_chimera/outputs/_dw39_smoke`
  - train_loss: `1.373855`
  - val_loss: `1.504337`
  - artifacts copied locally under `/Users/dongwoo/Documents/Codex/2026-04-25/38/anti_chimera/outputs/dw39_smoke/`
- Main run:
  - output dir: `/home/dongwoo39/projects/anti_chimera/outputs/lite3d_dw39`
  - step: `400`
  - train_loss: `0.35876058757305146`
  - val_loss: `0.07990326453000307`
  - artifacts copied locally under `/Users/dongwoo/Documents/Codex/2026-04-25/38/anti_chimera/outputs/dw39_lite3d/`
- Continued run:
  - output dir: `/home/dongwoo39/projects/anti_chimera/outputs/lite3d_dw39_long`
  - step: `1000`
  - train_loss: `0.15987199986043077`
  - val_loss: `0.060147817712277174`
  - low-t val loss remained worse than high-t val loss, so the final denoising region is still the weak point.
- Longer continuation:
  - output dir: `/home/dongwoo39/projects/anti_chimera/outputs/lite3d_dw39_4k`
  - step: `4000`
  - train_loss: `0.07056607975147101`
  - val_loss: `0.015582347754389048`
  - val_loss_low_t: `0.04178733844310045`
  - val_loss_high_t: `0.003729640007285135`
- EMA continuation:
  - output dir: `/home/dongwoo39/projects/anti_chimera/outputs/lite3d_dw39_ema5k`
  - step: `5000`
  - train_loss: `0.059272968608420344`
  - val_loss: `0.012727743247523904`
  - val_loss_low_t: `0.04164091870188713`
  - val_loss_high_t: `0.0037662205951554434`
  - checkpoint includes `ema_model`
- Sampling after EMA + x0 clamp:
  - output: `/home/dongwoo39/projects/anti_chimera/outputs/_tmp_ema5k_sample.gif`
  - visible result: still imperfect, but no longer pure noise; it now shows small object-like blobs and clearer color structure.
- Refinement demo:
  - command: `scripts/refine.py --start-timestep 50`
  - output: `/home/dongwoo39/projects/anti_chimera/outputs/_tmp_refine_t50.gif`
  - result: one-step / low-noise refinement produces a visibly structured frame montage instead of pure noise.

#### Current interpretation

- The training loop is healthy and the loss drops consistently.
- Pure sampling from full noise has improved from pure noise to weak but structured blobs after EMA + x0 clamping.
- The model clearly reconstructs low-noise inputs, so the useful inference path is now both refinement and modest full-noise generation.
- The remaining bottleneck is still high-noise chain quality and overall sample fidelity, not end-to-end execution.

### Experiment plan

- Start with the synthetic collision dataset on `backend=lite3d`.
- Verify that loss decreases and that sampled videos show structured motion instead of pure noise.
- If the lite path is healthy, keep CogVideoX as a future extension path rather than the first debugging target.
- For the dw39 continuation, prioritize improving the high-noise reverse chain if pure generation is still required.
- If more compute is available, continue from the EMA checkpoint and try a second phase with a slightly lower learning rate plus a longer sampling schedule.


## 2026-04-25 - Backbone / Data Prep Alignment

### Hypothesis
The current custom lite path should stay as smoke-only, while real training should reuse the pretrained CogVideoX stack and only train the anti-chimera conditioning path on a manifest built from real video data.

### Changes staged
- Added `scripts/prepare_viddata.py` to download `Databoost/VidData` from Hugging Face and emit train / val manifests plus CSV and metadata summaries.
- Added `configs/cogvideox_viddata.yaml` for the main real-data CogVideoX run.
- Added `configs/cogvideox_viddata_smoke.yaml` for a short verification run.
- Added `freeze_transformer` support to the CogVideoX wrapper so the backbone can stay frozen by default.
- Updated the docs and README to point real-data experiments to the VidData manifest path instead of the synthetic-only lite route.

### Next step
Install missing ML dependencies in the `jelly` conda environment, download VidData, and run a short smoke training pass on the new manifest to confirm the end-to-end real-data path is wired correctly.


## 2026-04-25 - Official CogVideoX LoRA Smoke

### Hypothesis
Official Diffusers CogVideoX LoRA training can run end-to-end on the VidData manifest if we keep the backbone stack intact, inject scene conditioning as a bias on prompt embeddings, and use a validation path that does not alter the transformer token contract.

### Changes
- Added `anti_chimera/trainer_cogvideox.py` with an official-stack CogVideoX LoRA trainer built on `Accelerator`, `CogVideoXPipeline`, `CogVideoXDPMScheduler`, and `peft` LoRA adapters.
- Routed `scripts/train.py` to the CogVideoX trainer when `model.backend == "cogvideox"`.
- Kept `ManifestVideoDataset` as the data bridge and used prompt-derived scene hints as a conditioning bias instead of appending extra tokens to the CogVideoX text sequence.
- Disabled cuDNN and NCCL P2P/IB on the RTX 4090 node to avoid backend init issues during smoke runs.
- Updated the CogVideoX smoke config to `17` frames and short `max_train_steps`.

### Smoke result
- Remote output dir: `/home/dongwoo39/projects/anti_chimera/outputs/cogvideox_viddata_smoke`
- Final metrics: `step 1 loss 0.8573`, `step 2 loss 0.6917`, `step 3 loss 0.5296`, `step 4 loss 0.2777`
- Final checkpoint: `/home/dongwoo39/projects/anti_chimera/outputs/cogvideox_viddata_smoke/final`
- Saved sample files: `/home/dongwoo39/projects/anti_chimera/outputs/cogvideox_viddata_smoke/samples/step_0004.png` and `/home/dongwoo39/projects/anti_chimera/outputs/cogvideox_viddata_smoke/samples/step_0004.gif`
- Local preview copy: `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_viddata_smoke/step_0004.png`
- Visual note: the 4-step smoke sample is still mostly colored noise, which is expected at this step budget, but the full training / validation / save loop now runs cleanly.

### Next step
- Run a longer continuation with the same official CogVideoX stack but fewer validation passes per step, then compare checkpointed samples against the smoke baseline.

### Continuation result
- Config: `configs/cogvideox_viddata_continuation.yaml`
- Remote output dir: `/home/dongwoo39/projects/anti_chimera/outputs/cogvideox_viddata_continuation`
- Final metrics: `step 1 loss 0.8573`, `step 4 loss 0.2771`, `step 8 loss 0.4821`, `step 12 loss 0.3151`
- Final checkpoint: `/home/dongwoo39/projects/anti_chimera/outputs/cogvideox_viddata_continuation/final`
- Local preview copy: `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_viddata_continuation/step_0012.png`
- Visual note: the 12-step sample is still mostly abstract banded structure, not semantic video yet. The official stack is correct, but the shortened smoke geometry is still too weak for meaningful generation.

## 2026-04-25 - Official CogVideoX VidData Sidecar Continuation

### Hypothesis
The official CogVideoX LoRA stack can run on the full VidData manifest with sidecar-based scene conditioning if training keeps the backbone intact, the real data manifest is enriched with per-video sidecars, and validation sampling is decoupled from the full training resolution to avoid preview OOM.

### Changes
- Added `scripts/prepare_viddata_sidecars.py` to derive per-video `tracks`, `depth`, and `visibility` sidecars from the real VidData videos.
- Updated `anti_chimera/data/manifest.py` so `.npz` sidecars are read by key instead of blindly loading the first array.
- Added `configs/cogvideox_viddata_official.yaml` for the higher-resolution continuation run.
- Reduced the validation preview resolution and frame count in `trainer_cogvideox.py` via `sampling.image_size` and `sampling.num_frames` overrides so preview sampling no longer OOMs at training resolution.

### Data prep result
- Enriched manifests:
  - `data/viddata_sidecar/train.jsonl` with `905` rows
  - `data/viddata_sidecar/val.jsonl` with `101` rows
- Sidecar shapes verified on smoke and full data:
  - `tracks`: `[25, 4, 4]`
  - `depth`: `[25, 256, 256]`
  - `visibility`: `[25, 4]`

### Training result
- Run dir: `outputs/cogvideox_viddata_official`
- Resume checkpoint: `outputs/cogvideox_viddata_continuation/final`
- Training steps: `60`
- Loss trend:
  - `step 1: 0.5579`
  - `step 2: 0.4138`
  - `step 30: 0.2699`
  - `step 60: 0.4032`
- Final checkpoint: `outputs/cogvideox_viddata_official/final`
- Final samples saved:
  - `outputs/cogvideox_viddata_official/samples/step_0060.png`
  - `outputs/cogvideox_viddata_official/samples/step_0060.gif`
- Local preview copy:
  - `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_viddata_official/step_0060.png`

### Interpretation
- The full official stack is now stable on the real VidData sidecar path.
- Validation sampling no longer OOMs after decoupling preview size from training size.
- The sample is still mostly abstract banded structure rather than semantic video, so the remaining bottleneck is sample quality, not pipeline execution.

### Next step
- Try a second continuation phase from `outputs/cogvideox_viddata_official/final` with a lower learning rate and a slightly longer run.
- If sample quality still stalls, revisit conditioning strength or the sidecar heuristic itself instead of changing the backbone stack.

## 2026-04-25 - Official CogVideoX Condition-Boost Probe

### Hypothesis
The remaining failure mode is that the sidecar bias is too weak. Increasing the scene bias scale should move the model away from generic banded texture and toward the real video structure.

### Change
- Updated `SceneConditionEncoder.forward()` to use `1.0 + sigmoid(gate)` instead of `sigmoid(gate)` so the sidecar bias is no longer heavily damped.

### Early result
- Config: `configs/cogvideox_viddata_official_boost.yaml`
- Resume checkpoint: `outputs/cogvideox_viddata_official/final`
- Step-20 preview still showed the same abstract banded structure as the lower-gain runs.
- Interpretation: simple gain scaling is not enough by itself; the bottleneck is likely the quality of the sidecar signal or the injection point, not just the amplitude.

### Next step
- Replace the motion-only sidecar heuristic with a stronger semantic signal, or inject scene conditioning deeper than prompt embedding bias.

## 2026-04-25 - 4GPU CogVideoX continuation fix

### Hypothesis
The 4GPU run was failing because the official CogVideoX LoRA trainer was being launched through `accelerate` with a DDP-wrapped model, while the PEFT load/save and config access code still assumed an unwrapped module.

### Changes
- Added a repo-root `sys.path` bootstrap to `scripts/train.py` so `accelerate launch` can import `anti_chimera` without needing manual `PYTHONPATH` setup.
- Updated `anti_chimera/trainer_cogvideox.py` to unwrap the DDP model before calling PEFT load/save helpers.
- Updated the validation sampling path to read `transformer_base.config` instead of the wrapped module config.

### Verification
- `vic_repro` import check passed for `anti_chimera` and `diffusers.CogVideoXPipeline`.
- `accelerate launch --num_processes 4` now starts all 4 ranks on `dw39`.
- Training reached live steps with logged losses:
  - `step 1: 0.4719`
  - `step 2: 0.3381`

### Status
- The 4GPU continuation is now running instead of failing at import or DDP/PEFT initialization.
- Next checkpoint is to let it reach the first scheduled sample at `step 20` and inspect the generated preview.


## 2026-04-25 semantic full run prep
- full semantic sidecar output dir: `data/viddata_sidecar_semantic`
- full training config prepared: `configs/cogvideox_viddata_semantic_full.yaml`
- target: 80 steps, sample every 20, checkpoint every 40


## 2026-04-25 semantic full run update
- Full semantic sidecar generation completed with DETR-batched frame inference.
- Semantic sidecars produced:
  - train: 905 rows
  - val: 101 rows
  - mean visibility: train 0.6834, val 0.7230
- Completed `cogvideox_viddata_semantic_full` continuation to 80 steps.
- Observations from samples:
  - `step_0020`: abstract tiled color blocks, no semantic alignment.
  - `step_0040`: still abstract, no object-level structure.
  - `step_0060`: slightly more coherent color bands, but still far from target structure.
  - `step_0080`: still abstract stripe/grid artifacts; target structure not recovered.
- Conclusion: 80-step run is insufficient for meaningful semantic grounding.
- Next run launched:
  - config: `configs/cogvideox_viddata_semantic_full_long.yaml`
  - resume: `outputs/cogvideox_viddata_semantic_full/final`
  - changes:
    - `conditioning_mode: hybrid`
    - `scene_prompt_scale: 0.1`
    - `latent_control_scale: 0.5`
    - `learning_rate: 1e-5`
    - `max_train_steps: 400`
    - `sample_every: 100`
    - `checkpointing_steps: 200`

## 2026-04-26 semantic-plus pipeline update
- Added  to generate  sidecars from VidData.
- Switched  and  to consume the new semantic-plus keys.
- Updated  to collate , add reference-frame conditioning, and partially load mismatched resume checkpoints.
- Added smoke/full configs:
  - 
  - 
- Added  as a wrapper for VBench / Video-Bench CLI execution.
- Smoke sidecar generation succeeded on 1 train + 1 val sample.
- Smoke 4GPU training on  completed 8 steps and saved  and  samples.
- Current smoke samples remain abstract / stripe-like; conditioning path is now verified end-to-end, but quality is still not meaningful.
- Full semantic-plus sidecar generation is running in the background at .


## 2026-04-26 semantic-plus pipeline update
- Added  to generate  sidecars from VidData.
- Switched  and  to consume the new semantic-plus keys.
- Updated  to collate , add reference-frame conditioning, and partially load mismatched resume checkpoints.
- Added smoke/full configs:
  - 
  - 
- Added  as a wrapper for VBench / Video-Bench CLI execution.
- Smoke sidecar generation succeeded on 1 train + 1 val sample.
- Smoke 4GPU training on  completed 8 steps and saved  and  samples.
- Current smoke samples remain abstract / stripe-like; conditioning path is now verified end-to-end, but quality is still not meaningful.
- Full semantic-plus sidecar generation is running in the background at .


## 2026-04-26 semantic-plus pipeline update
- Added scripts/prepare_viddata_sidecars_semantic_plus.py to generate tracks, masks, depth, visibility, flow, and occlusion sidecars from VidData.
- Switched anti_chimera/data/scene_hint.py and anti_chimera/data/manifest.py to consume the new semantic-plus keys.
- Updated anti_chimera/trainer_cogvideox.py to collate masks, flow, and occlusion, add reference-frame conditioning, and partially load mismatched resume checkpoints.
- Added smoke/full configs for semantic-plus training.
- Added scripts/evaluate_video_benchmarks.py as a wrapper for VBench and Video-Bench CLI execution.
- Smoke sidecar generation succeeded on one train and one val sample.
- Smoke 4GPU training on outputs/cogvideox_viddata_semantic_plus_smoke completed 8 steps and saved step_0004 and step_0008 samples.
- Current smoke samples remain abstract and stripe-like; conditioning path is now verified end-to-end, but quality is still not meaningful.
- Full semantic-plus sidecar generation is running in the background at data/viddata_sidecar_semantic_plus.


## 2026-04-26 benchmark loop update
- VBench now runs on the smoke samples and produced a subject consistency score of 0.8106206469237804 for the staged step_0004 and step_0008 videos.
- Video-Bench now has the correct staging layout and prompt mapping for custom_nonstatic evaluation; the temporal_consistency run is still executing.
## 2026-04-26 - Meaningful Video Quality Recovery Loop

### Hypothesis
- The full T2V continuation was producing tiled / abstract samples because the recovery loop was attacking task scale before proving the official objective and first-frame reference task.
- The most likely recoverable path is: official CogVideoX objective alignment -> first-frame I2V reconstruction -> sidecar ablation -> larger continuation.

### Code changes
- Aligned `anti_chimera/trainer_cogvideox.py` around `objective_mode: official_cogvideox` and kept the official CogVideoX target form: scheduler velocity converted with clean latent target.
- Fixed the custom latent-control path so sidecar residuals modify the model input but do not contaminate the scheduler `get_velocity(...)` reference latent.
- Added first-frame I2V reference support for CogVideoX I2V-style transformers where `in_channels == 2 * latent_channels`.
- Added `precompute_latents: true` for CogVideoX 5B I2V probes so VAE/text latents are cached before the transformer is moved to GPU.
- Added bf16-safe sampling/training fixes:
  - sampler uses `scheduler.scale_model_input(...)`
  - sampler autocast follows `weight_dtype`
  - `SceneConditionEncoder` pooling output is cast back to module dtype
  - cached I2V image latents now use the configured official-style image noise sigma
- Hardened `ManifestVideoDataset` so sidecar tracks/depth/visibility/masks/flow/occlusion are time-resampled and spatially resized to match the requested clip schedule.
- Added `scripts/prepare_curriculum_manifest.py` for `curated8`, `curated32`, and `curated256` manifests.
- Added `scripts/sample_cogvideox_i2v_lora.py` for official `CogVideoXImageToVideoPipeline` sampling with CPU offload, LoRA loading, first-frame reference, PNG grid, and GIF output.
- Added `scripts/sample_cogvideox_sidecar_ablation.py` to reuse existing CogVideoX components for direct sidecar ablation. The 5B direct latent-control sampler currently exceeds 24GB even with sequential CPU offload, so it is recorded as the next engineering bottleneck rather than treated as passed.
- Extended `scripts/evaluate_local_quality.py` so it can score `*_sample_grid.png` against `target_grid.png` with configurable grid columns and label height.
- Added configs:
  - `configs/cogvideox_quality_probe_t2v_overfit8.yaml`
  - `configs/cogvideox_quality_probe_i2v_overfit8.yaml`
  - `configs/cogvideox_quality_probe_i2v_overfit8_128.yaml`
  - `configs/cogvideox_quality_probe_i2v15_overfit8_128.yaml`
  - `configs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200.yaml`

### Phase 0 snapshot
- Remote repo: `/home/dongwoo39/projects/anti_chimera`
- Server: `dw39`
- GPUs after runs: idle, no active compute process.
- Stale Video-Bench / VBench processes: none active during final snapshot.
- Main successful output dir: `outputs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200`

### Phase 1 objective probe
- T2V overfit8 run completed with finite loss but failed visually:
  - output: `outputs/cogvideox_quality_probe_t2v_overfit8`
  - final step: `40`
  - final loss: about `0.235`
  - visual result: still tiled/color-block structure, not semantic video
- Interpretation: T2V LoRA on tiny low-res probes is not a useful first quality gate. The official objective is runnable, but the task is too hard without reference conditioning.

### Phase 2 first-frame I2V probe
- `THUDM/CogVideoX-5b-I2V` at reduced resolution was not usable:
  - 256/25 attempt hit VAE OOM
  - 128/17 attempt hit fixed learned positional embedding resolution constraints
- Switched to `zai-org/CogVideoX1.5-5B-I2V`, which uses rotary positional embeddings and supports the reduced probe schedule.
- fp16 training produced NaN loss, but bf16 fixed it.
- 20-step hybrid I2V smoke:
  - output: `outputs/cogvideox_quality_probe_i2v15_overfit8_128`
  - final checkpoint saved
  - loss stayed finite, roughly `0.18-0.71`
  - official I2V sampling preserved the first-frame scene for early frames but collapsed in later frames at 128.

### Phase 2 recovery run
- Ran reference-only official I2V LoRA overfit:
  - config: `configs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200.yaml`
  - command: `CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 scripts/train.py --config configs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200.yaml`
  - train set: `curated8`
  - model: `zai-org/CogVideoX1.5-5B-I2V`
  - LoRA rank/alpha: `16/16`
  - steps: `200`
  - precision: `bf16`
  - lr: `1e-4`
  - sidecar scales: disabled for this gate (`scene_prompt_scale=0`, `latent_control_scale=0`)
- Loss improved from early noisy values around `0.30-0.72` to repeated late values around `0.098-0.19`.
- Final checkpoint:
  - `outputs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200/final`

### Phase 2 visual gate
- Official I2V CPU-offload sampler generated three 256px sample grids and GIFs:
  - `outputs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200/i2v_samples_s0_256/lora_sample_grid.png`
  - `outputs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200/i2v_samples_s1_256/lora_sample_grid.png`
  - `outputs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200/i2v_samples_s2_256/lora_sample_grid.png`
- Local copies:
  - `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200/i2v_samples_s0_256/`
  - `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200/i2v_samples_s1_256/`
  - `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200/i2v_samples_s2_256/`
- Summary PNG:
  - remote: `outputs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200/meaningful_reference_gate_summary.png`
  - local: `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200/meaningful_reference_gate_summary.png`
- Visual interpretation:
  - PASS for reference gate: scene layout, main people, object/background separation are preserved across all 3 checked clips.
  - PARTIAL for meaningful video: outputs are semantically structured but too static/repetitive; temporal motion is weak.
  - NOT PASSED for conditioning gate: direct latent sidecar influence is not yet proven.

### Local metrics
- `s0_256`:
  - base target MSE: `0.2242`, temporal_change: `0.0617`
  - ref-only LoRA target MSE: `0.2603`, temporal_change: `0.0371`
- `s1_256` LoRA:
  - target MSE: `0.2347`
  - temporal_change: `0.0380`
- `s2_256` LoRA:
  - target MSE: `0.2014`
  - temporal_change: `0.0352`
- Interpretation: the metric confirms the current model is conservative/static; the visual gain is structural preservation, not motion richness.

### Phase 3 sidecar ablation status
- Added a direct sidecar ablation sampler using the existing CogVideoX components and trained `scene_encoder`, `latent_controller`, and `reference_encoder`.
- Attempts on the 5B I2V transformer at 128 and 64 resolution failed with CUDA OOM during manual transformer forward, even with `enable_model_cpu_offload` and `enable_sequential_cpu_offload`.
- Root cause: direct custom latent-control sampling bypasses the official I2V pipeline's memory-efficient internals enough that the 5B transformer still needs almost the full 24GB on one GPU.
- Current decision: do not claim sidecar gate. The next high-impact engineering task is to move latent-control injection into an official `CogVideoXImageToVideoPipeline` subclass/callback path or run the direct sampler with model parallel/offload that actually shards transformer blocks.

### Current conclusion
- The project recovered a reproducible path to meaningful, non-tiled video structure using official CogVideoX1.5 I2V + first-frame reference + 200-step overfit.
- The previous long T2V continuation was not the right next step.
- The immediate bottleneck is no longer objective instability; it is sidecar/control injection under memory constraints and temporal motion diversity.

### Next experiment
- Implement the sidecar branch inside an official `CogVideoXImageToVideoPipeline` subclass, not a fully manual sampler.
- Keep the 200-step reference-only checkpoint as the quality baseline.
- Run prompt/reference/latent-control A/B only after the sidecar sampler can produce 128 or 256 PNG grids without exceeding 24GB.

## 2026-04-26 - Hybrid Sidecar Conditioning Gate

### Hypothesis
- The previous sidecar path was too weak or too memory-heavy because it either compressed sidecars into global prompt bias or bypassed the official CogVideoX I2V pipeline memory path.
- A minimal official-pipeline subclass that keeps CogVideoX I2V internals and injects only prompt-bias + latent residual should provide a measurable sidecar effect without custom sampler OOM.

### Code changes
- Added `scripts/sample_cogvideox_i2v_sidecar_ablation.py`.
  - Subclasses `CogVideoXImageToVideoPipeline`.
  - Reuses official prompt encoding, I2V image-latent preparation, scheduler step, CPU offload, and VAE decode.
  - Adds only two controlled hooks:
    - `sidecar_prompt_bias` on positive prompt embeddings.
    - `sidecar_latent_residual` on the denoising latent input before image-latent concat.
- Added `configs/cogvideox_quality_probe_i2v15_sidecar_prompt_overfit8_128_200.yaml`.
  - Resumes from the reference-only 200-step checkpoint.
  - Uses prompt-side scene conditioning only.
- Added `configs/cogvideox_quality_probe_i2v15_hybrid_sidecar_overfit8_128_200.yaml`.
  - Resumes from the reference-only 200-step checkpoint.
  - Uses `scene_prompt_scale=0.10`, `latent_control_scale=0.15`, `latent_control_warmup_steps=25`, and `latent_control_gate_init=-3.0`.
- Patched `anti_chimera/models/control.py` for bf16-safe condition/reference projection dtypes.

### Prompt-only sidecar result
- Run: `outputs/cogvideox_quality_probe_i2v15_sidecar_prompt_overfit8_128_200`
- Result: prompt-bias-only was not enough.
  - `s0` improved target MSE/L1 and tile repetition.
  - `s1` and `s2` were neutral or slightly worse on target metrics.
- Interpretation: global text bias is not a sufficient proof of scene-conditioning influence.

### Hybrid sidecar run
- Run: `outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_overfit8_128_200`
- Training:
  - model: `zai-org/CogVideoX1.5-5B-I2V`
  - base checkpoint: `outputs/cogvideox_quality_probe_i2v15_refonly_overfit8_128_200/final`
  - 4 GPUs
  - 200 steps
  - bf16
  - late losses stayed finite, with late values around `0.08-0.34`
- Sampling:
  - official-pipeline sidecar sampler completed at 256px for `s0`, `s1`, and `s2`.
  - variants: `no_sidecar`, `latent_control`, `full_sidecar`
  - no OOM after moving the sidecar branch into the official I2V pipeline subclass.

### Hybrid sidecar metrics
- Summary files:
  - remote PNG: `outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_overfit8_128_200/hybrid_sidecar_gate_summary.png`
  - remote JSON: `outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_overfit8_128_200/hybrid_sidecar_gate_metrics_summary.json`
  - local PNG: `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_overfit8_128_200/hybrid_sidecar_gate_summary.png`
  - local JSON: `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_overfit8_128_200/hybrid_sidecar_gate_metrics_summary.json`
- Aggregate `full_sidecar` vs `no_sidecar`:
  - target MSE: `-1.74%` mean relative delta, improved `3/3`
  - target L1: `-1.09%` mean relative delta, improved `3/3`
  - temporal change: `+8.31%` mean relative delta, improved `3/3`
  - tile repetition: `-0.06%` mean relative delta, improved `1/3`
- Per-sample:
  - `s0`: MSE `0.2580 -> 0.2561`, L1 `0.3917 -> 0.3897`, temporal `0.0353 -> 0.0363`, tile `0.5681 -> 0.5684`
  - `s1`: MSE `0.2325 -> 0.2272`, L1 `0.3748 -> 0.3700`, temporal `0.0348 -> 0.0412`, tile `0.5511 -> 0.5479`
  - `s2`: MSE `0.2007 -> 0.1964`, L1 `0.3491 -> 0.3440`, temporal `0.0387 -> 0.0402`, tile `0.5433 -> 0.5453`

### Interpretation
- Conditioning gate is now partially passed.
  - Positive: full sidecar beats no-sidecar on target adherence and temporal change in all 3 checked probes.
  - Positive: strongest qualitative/metric gain is `s1`, where target adherence, motion, and tile repetition all improve.
  - Limitation: tile repetition / chimera suppression is not yet robust across all probes; only `1/3` improved on the current tile metric.
- This is evidence that sidecar influences generation, but not yet enough for the final project goal. The next phase should not run a blind long continuation yet.

### Next experiment
- Keep `configs/cogvideox_quality_probe_i2v15_hybrid_sidecar_overfit8_128_200.yaml` as the current best sidecar baseline.
- Add a stronger object/mask-aware local metric because current tile repetition is too blunt.
- Run a scale sweep on `latent_control_scale` around `0.10`, `0.20`, and `0.30` using the same 3 sample indices/seeds.
- If at least `2/3` probes improve on target adherence, temporal change, and repetition/chimera metric, expand to `curated32`.

## 2026-04-26 - Sidecar Scale Sweep and Mask-Aware Chimera Proxy

### Hypothesis
- The `0.15` latent control scale was too conservative, so sidecar influence was measurable in target L1/MSE but not strong enough on the repetition/chimera proxy.
- The previous global tile metric is too blunt for chimera: if chimera is object blending, foreground object-mask metrics should be more diagnostic than whole-contact-sheet similarity.

### Execution
- Ran inference-only latent scale sweep from `outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_overfit8_128_200/final`.
  - scales: `0.10`, `0.20`, `0.30`
  - samples: `s0`, `s1`, `s2`
  - variants: `no_sidecar`, `full_sidecar`
  - result file: `outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_overfit8_128_200/scale_sweep_metrics_summary.json`
- Best scale by aggregate score was `0.30`, but it still improved tile repetition on only `1/3` probes.
  - scale `0.30` mean relative deltas:
    - target MSE: `-1.73%`, improved `3/3`
    - target L1: `-1.09%`, improved `3/3`
    - temporal change: `+8.30%`, improved `3/3`
    - tile repetition: `-0.07%`, improved `1/3`

### Stronger sidecar continuation
- Added `configs/cogvideox_quality_probe_i2v15_hybrid_sidecar_v2_overfit8_128_160.yaml`.
  - resume: `outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_overfit8_128_200/final`
  - output: `outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_v2_overfit8_128_160`
  - 4 GPUs
  - 160 steps
  - `scene_prompt_scale=0.15`
  - `latent_control_scale=0.35`
  - `latent_control_gate_init=-1.5`
  - `learning_rate=4e-5`
- Training completed without NaN/OOM.
  - late losses stayed finite, mostly around `0.06-0.27` with occasional spikes.
  - final checkpoint saved with LoRA and sidecar modules.

### V2 whole-image result
- Summary files:
  - remote PNG: `outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_v2_overfit8_128_160/v2_sidecar_gate_summary.png`
  - remote JSON: `outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_v2_overfit8_128_160/v2_sidecar_gate_metrics_summary.json`
  - local PNG: `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_v2_overfit8_128_160/v2_sidecar_gate_summary.png`
  - local JSON: `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_v2_overfit8_128_160/v2_sidecar_gate_metrics_summary.json`
- Aggregate `full_sidecar` vs `no_sidecar`:
  - target MSE: `-1.52%`, improved `3/3`
  - target L1: `-1.02%`, improved `3/3`
  - temporal change: `+8.85%`, improved `3/3`
  - tile repetition: `-0.17%`, improved `1/3`
- Interpretation:
  - Sidecar influence is stable.
  - Whole-image tile repetition still does not robustly pass; the model remains too static/repetitive.

### Mask-aware metric implementation
- Extended `scripts/evaluate_local_quality.py` with optional sidecar mask evaluation:
  - `--manifest`
  - `--root-dir`
  - `--sample-index`
  - `--data-num-frames`
  - `--data-image-size`
  - `--max-objects`
- New metrics:
  - `mask_foreground_l1`
  - `mask_foreground_mse`
  - `mask_background_l1`
  - `mask_contrast_error`
  - `mask_foreground_temporal_change`
- This uses the existing `ManifestVideoDataset` masks and compares generated contact sheets against target frames in object foreground regions.

### V2 mask-aware result
- Summary files:
  - remote PNG: `outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_v2_overfit8_128_160/v2_sidecar_mask_gate_summary.png`
  - remote JSON: `outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_v2_overfit8_128_160/v2_sidecar_mask_gate_metrics_summary.json`
  - local PNG: `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_v2_overfit8_128_160/v2_sidecar_mask_gate_summary.png`
  - local JSON: `/Users/dongwoo/Documents/Codex/2026-04-25/users-dongwoo-documents-codex-2026-04/outputs/cogvideox_quality_probe_i2v15_hybrid_sidecar_v2_overfit8_128_160/v2_sidecar_mask_gate_metrics_summary.json`
- Aggregate `full_sidecar` vs `no_sidecar`:
  - mask foreground L1: `-4.45%`, improved `3/3`
  - mask foreground MSE: `-6.49%`, improved `3/3`
  - mask background L1: `-5.62%`, improved `2/3`
  - temporal change: `+8.85%`, improved `3/3`
  - tile repetition: `-0.17%`, improved `1/3`
- Per-sample foreground metrics:
  - `s0`: mask FG L1 `0.2141 -> 0.2090`, mask FG MSE `0.0953 -> 0.0915`
  - `s1`: mask FG L1 `0.1946 -> 0.1888`, mask FG MSE `0.0776 -> 0.0748`
  - `s2`: mask FG L1 `0.1438 -> 0.1323`, mask FG MSE `0.0383 -> 0.0337`

### Conclusion
- Evidence obtained: sidecar improves object foreground coherence against no-condition in all 3 probes.
- This is the strongest current evidence that sidecar reduces object-region chimera / blending, because the metric uses sidecar object masks instead of whole-image averages.
- Remaining limitation: temporal foreground motion and whole-image repetition are still weak. The model is coherent but too static.

### Next experiment
- Keep V2 as the current best checkpoint for sidecar evidence.
- Add a real temporal-object metric using mask centroid displacement / flow alignment rather than contact-sheet tile similarity.
- Then expand from `curated8` to `curated32` only if foreground object metrics remain positive.
