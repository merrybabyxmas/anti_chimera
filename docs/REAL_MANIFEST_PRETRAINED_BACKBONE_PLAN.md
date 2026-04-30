# Real Manifest + Pretrained Backbone Execution Plan

This document defines the **actual priority order** for the project when the final objective is:

> achieve meaningful anti-chimera improvement on **real video manifests** with a **pretrained video backbone**.

The repository still contains synthetic and lightweight paths, but they are no longer the final target. They are now treated as supporting tools for diagnosis, smoke tests, and planner pretraining.

---

## 1. Final target

The final target is **not**:

- perfect 3D reconstruction,
- synthetic-only gains,
- or an over-engineered planner.

The final target **is**:

1. keep a pretrained backbone intact,
2. add a small anti-chimera conditioning path,
3. use minimal scene structure,
4. improve multi-entity collision behavior on real data.

The intended full real-data system is:

```text
prompt
→ learned planner
→ minimal scene hint
→ pretrained video backbone
→ video
```

where the minimal scene hint is deliberately restricted to:

- slot box grounding
- depth bins
- slot visibility
- overlap map

---

## 2. What must be true before claiming success

The repository should only claim real-backbone success after the following three results are established.

### Result A. Oracle-conditioned real-backbone improvement

The pretrained backbone must improve when it receives real/pseudo structural sidecars directly.

This answers:

> does structured scene prior actually help the pretrained backbone reduce chimera artifacts?

### Result B. Learned-planner prompt-only improvement

Prompt-only validation using the learned planner must improve over the plain prompt-only baseline.

This answers:

> can the learned planner recover enough useful structure from text to transfer part of the oracle gain into real prompt-only generation?

### Result C. Minimal scene hint is sufficient

The improvement should appear with the minimal scene hint itself. This matters because the final real pipeline must stay robust to noisy pseudo-labels.

---

## 3. Priority order

The correct priority order is:

### Priority 1. Build a real manifest with sidecars

Each clip should ideally have:

- `video_path`
- `caption`
- `tracks_path`
- `depth_path`
- `visibility_path`

Optional but allowed:

- `masks_path`
- `flow_path`
- `occlusion_path`

The main real experiment should **not** wait for perfect masks or dense flow. Tracks, depth, and visibility already support the minimal anti-chimera path.

### Priority 2. Validate the manifest quality before training

The manifest must be checked for:

- missing sidecars
- shape mismatches
- invalid frame counts
- clips that cannot be read

This repo now includes a dedicated checker script for that purpose.

### Priority 3. Oracle-conditioned pretrained-backbone run

Before spending effort on a better planner, run the pretrained backbone with direct sidecar conditioning and check whether the oracle-conditioned validation improves.

### Priority 4. Train the learned planner on the same real sidecars

The planner should imitate the real sidecars, not an idealized world model.

### Priority 5. Run prompt-only validation with the learned planner

The learned planner should then be plugged into prompt-only validation and compared against the oracle-conditioned upper bound.

---

## 4. Recommended concrete execution order

### Step 1. Build the manifest

Use `scripts/build_manifest.py` to produce:

- `data/train.jsonl`
- `data/val.jsonl`

### Step 2. Check the manifest and sidecars

```bash
python scripts/check_manifest_sidecars.py \
  --manifest data/train.jsonl \
  --root-dir . \
  --num-frames 12 \
  --image-size 64 \
  --max-objects 3
```

Do the same for validation.

### Step 3. Train the real-data planner

```bash
python scripts/train_planner.py --config configs/planner_real_template.yaml
```

### Step 4. Run the real pretrained-backbone experiment

For CogVideoX 2B:

```bash
python scripts/train.py --config configs/cogvideox_2b.yaml
```

For CogVideoX 5B:

```bash
python scripts/train.py --config configs/cogvideox_5b.yaml
```

### Step 5. Evaluate both oracle-conditioned and planner-conditioned prompt-only generation

```bash
python scripts/evaluate_real_manifest.py \
  --config configs/cogvideox_2b.yaml \
  --checkpoint outputs/cogvideox_2b/checkpoints/last.pt \
  --limit 32
```

This will report averaged oracle-conditioned and prompt-only metrics separately.

---

## 5. Why the minimal scene hint is now the default

Real pseudo-labels are noisy.

If the condition becomes too dense or too complicated, the pretrained backbone may either:

- ignore it,
- overfit to noise,
- or become unstable.

That is why the current default is **minimal**, not full.

The real objective is not to give the backbone every possible signal. The objective is to give it the **smallest reliable structural prior** that helps preserve identity under collision and occlusion.

---

## 6. What is intentionally deferred

The following are explicitly deferred until the real minimal path already works:

- full 3D scene modeling
- aggressive attention surgery
- end-to-end planner-generator joint finetuning
- large dense-prior stacks as the default path
- synthetic-only claims

---

## 7. Success criterion

A successful real-data result should look like this:

1. oracle-conditioned validation beats the plain pretrained backbone baseline,
2. learned-planner prompt-only validation recovers a meaningful portion of that gain,
3. the gain is strongest on collision-heavy and overlap-heavy prompts,
4. the improvement appears without abandoning the pretrained backbone prior.

That is the correct success definition for this repository.
