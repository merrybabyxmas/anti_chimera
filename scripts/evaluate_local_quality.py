from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _basic_metrics(sample: np.ndarray, target: np.ndarray | None, columns: int | None, label_height: int) -> dict:
    metrics: dict[str, float] = {}
    if target is not None and target.shape != sample.shape:
        target = np.asarray(
            Image.fromarray((target * 255.0).astype(np.uint8)).resize(
                (sample.shape[1], sample.shape[0]),
                resample=Image.BICUBIC,
            ),
            dtype=np.float32,
        ) / 255.0
    if target is not None:
        mse = float(np.mean((sample - target) ** 2))
        metrics["target_mse"] = mse
        metrics["target_psnr"] = float(99.0 if mse <= 1e-12 else -10.0 * math.log10(mse))
        metrics["target_l1"] = float(np.mean(np.abs(sample - target)))
    metrics["sample_std"] = float(sample.std())
    metrics["sample_mean"] = float(sample.mean())
    metrics["tile_repetition"] = _tile_repetition_score(sample, columns=columns, label_height=label_height)
    metrics["temporal_change"] = _temporal_change_score(sample, columns=columns, label_height=label_height)
    return metrics


def _mask_metrics(
    sample: np.ndarray,
    target: np.ndarray | None,
    masks: list[np.ndarray],
    columns: int | None,
    label_height: int,
) -> dict:
    if target is None or not masks:
        return {}
    sample_frames = _split_contact_sheet(sample, columns=columns, label_height=label_height)
    target_frames = _split_contact_sheet(target, columns=columns, label_height=label_height)
    count = min(len(sample_frames), len(target_frames), len(masks))
    if count == 0:
        return {}

    fg_l1 = []
    fg_mse = []
    bg_l1 = []
    contrast_err = []
    fg_temporal = []
    prev_fg_sample = None
    for i in range(count):
        s = sample_frames[i]
        t = target_frames[i]
        m = masks[i]
        if m.shape[:2] != s.shape[:2]:
            m = np.asarray(
                Image.fromarray((m * 255.0).astype(np.uint8)).resize(
                    (s.shape[1], s.shape[0]),
                    resample=Image.BILINEAR,
                ),
                dtype=np.float32,
            ) / 255.0
        m = np.clip(m[..., None], 0.0, 1.0)
        fg_weight = float(m.mean())
        if fg_weight < 1e-4:
            continue
        bg = 1.0 - m
        bg_weight = float(bg.mean())
        diff = s - t
        fg_l1.append(float(np.sum(np.abs(diff) * m) / (np.sum(m) * 3.0 + 1e-8)))
        fg_mse.append(float(np.sum((diff ** 2) * m) / (np.sum(m) * 3.0 + 1e-8)))
        if bg_weight > 1e-4:
            bg_l1.append(float(np.sum(np.abs(diff) * bg) / (np.sum(bg) * 3.0 + 1e-8)))
            s_fg = float(np.sum(s * m) / (np.sum(m) * 3.0 + 1e-8))
            s_bg = float(np.sum(s * bg) / (np.sum(bg) * 3.0 + 1e-8))
            t_fg = float(np.sum(t * m) / (np.sum(m) * 3.0 + 1e-8))
            t_bg = float(np.sum(t * bg) / (np.sum(bg) * 3.0 + 1e-8))
            contrast_err.append(abs((s_fg - s_bg) - (t_fg - t_bg)))
        fg_sample = s * m
        if prev_fg_sample is not None:
            fg_temporal.append(float(np.sum(np.abs(fg_sample - prev_fg_sample) * m) / (np.sum(m) * 3.0 + 1e-8)))
        prev_fg_sample = fg_sample

    metrics = {}
    if fg_l1:
        metrics["mask_foreground_l1"] = float(np.mean(fg_l1))
        metrics["mask_foreground_mse"] = float(np.mean(fg_mse))
    if bg_l1:
        metrics["mask_background_l1"] = float(np.mean(bg_l1))
    if contrast_err:
        metrics["mask_contrast_error"] = float(np.mean(contrast_err))
    if fg_temporal:
        metrics["mask_foreground_temporal_change"] = float(np.mean(fg_temporal))
    return metrics


def _split_contact_sheet(image: np.ndarray, columns: int | None = None, label_height: int = 0) -> list[np.ndarray]:
    if label_height > 0 and image.shape[0] > label_height:
        image = image[label_height:]
    height, width = image.shape[:2]
    cols = columns or (4 if width % 4 == 0 else 1)
    tile_w = width // cols
    rows = max(1, height // tile_w)
    tile_h = height // rows
    frames = []
    for row in range(rows):
        for col in range(cols):
            frame = image[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w]
            if frame.size:
                frames.append(frame)
    return frames


def _tile_repetition_score(image: np.ndarray, columns: int | None = None, label_height: int = 0) -> float:
    frames = _split_contact_sheet(image, columns=columns, label_height=label_height)
    if len(frames) < 2:
        return 0.0
    vectors = []
    for frame in frames:
        small = frame[:: max(1, frame.shape[0] // 32), :: max(1, frame.shape[1] // 32)]
        vec = small.reshape(-1).astype(np.float32)
        vec = vec - vec.mean()
        norm = np.linalg.norm(vec) + 1e-8
        vectors.append(vec / norm)
    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(float(np.dot(vectors[i], vectors[j])))
    return float(np.mean(sims)) if sims else 0.0


def _temporal_change_score(image: np.ndarray, columns: int | None = None, label_height: int = 0) -> float:
    frames = _split_contact_sheet(image, columns=columns, label_height=label_height)
    if len(frames) < 2:
        return 0.0
    diffs = [np.mean(np.abs(frames[i + 1] - frames[i])) for i in range(len(frames) - 1)]
    return float(np.mean(diffs))


def _load_union_masks(
    manifest_path: Path,
    root_dir: Path,
    sample_index: int,
    num_frames: int,
    image_size: int,
    max_objects: int,
) -> list[np.ndarray]:
    from anti_chimera.data.manifest import ManifestVideoDataset

    dataset = ManifestVideoDataset(
        manifest_path=str(manifest_path),
        root_dir=str(root_dir),
        num_frames=num_frames,
        image_size=image_size,
        max_objects=max_objects,
    )
    item = dataset[sample_index % len(dataset)]
    masks = item.get("masks")
    if masks is None:
        return []
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks, dtype=np.float32)
    if masks.ndim != 4:
        return []
    # Dataset masks are [T, O, H, W]. Use union foreground over visible objects.
    union = np.clip(masks.max(axis=1), 0.0, 1.0)
    return [union[i] for i in range(union.shape[0])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--columns", type=int, default=None)
    parser.add_argument("--label-height", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--root-dir", type=Path, default=Path("."))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--data-num-frames", type=int, default=13)
    parser.add_argument("--data-image-size", type=int, default=128)
    parser.add_argument("--max-objects", type=int, default=4)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    results = []
    sample_paths = []
    masks = []
    if args.manifest is not None:
        masks = _load_union_masks(
            manifest_path=args.manifest,
            root_dir=args.root_dir,
            sample_index=args.sample_index,
            num_frames=args.data_num_frames,
            image_size=args.data_image_size,
            max_objects=args.max_objects,
        )
    for sample_path in sorted(sample_dir.glob("step_*.png")):
        if not sample_path.name.endswith("_target.png"):
            stem = sample_path.stem
            step = stem.split("_")[1] if "_" in stem else stem
            sample_paths.append((sample_path, sample_dir / f"step_{step}_target.png"))
    target_grid = sample_dir / "target_grid.png"
    if target_grid.exists():
        for sample_path in sorted(sample_dir.glob("*_sample_grid.png")):
            sample_paths.append((sample_path, target_grid))

    for sample_path, target_path in sample_paths:
        sample = _load_image(sample_path)
        target = _load_image(target_path) if target_path.exists() else None
        row = {"sample": str(sample_path), "target": str(target_path) if target_path.exists() else None}
        row.update(_basic_metrics(sample, target, columns=args.columns, label_height=args.label_height))
        if masks:
            row.update(_mask_metrics(sample, target, masks, columns=args.columns, label_height=args.label_height))
        results.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump({"num_samples": len(results), "samples": results}, f, ensure_ascii=False, indent=2)
    print(f"wrote {output} with {len(results)} samples")


if __name__ == "__main__":
    main()
