from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _split_grid(image: np.ndarray, columns: int = 7, label_height: int = 26) -> list[np.ndarray]:
    if label_height > 0:
        image = image[label_height:]
    h, w = image.shape[:2]
    tile_w = w // columns
    rows = max(1, math.ceil(h / tile_w))
    frames = []
    for row in range(rows):
        y1 = row * tile_w
        y2 = min(h, y1 + tile_w)
        if y2 <= y1:
            continue
        for col in range(columns):
            x1 = col * tile_w
            x2 = min(w, x1 + tile_w)
            if x2 <= x1:
                continue
            tile = image[y1:y2, x1:x2]
            if tile.shape[0] == tile_w and tile.shape[1] == tile_w:
                frames.append(tile)
    return frames


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    im = Image.fromarray((mask * 255.0).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _masked_mean_rgb(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None]
    denom = float(m.sum() * 3.0)
    if denom <= 1e-6:
        return np.full(3, 0.5, dtype=np.float32)
    return (frame * m).sum(axis=(0, 1)) / (m.sum() + 1e-8)


def _rgb_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.sqrt(3.0))


def _score_variant(sample_grid: Path, planned_path: Path, spec: dict, columns: int, label_height: int) -> dict:
    frames = _split_grid(_load_image(sample_grid), columns=columns, label_height=label_height)
    planned = torch.load(planned_path, map_location="cpu")
    masks = planned["masks"].float().numpy()[:, :2]
    rgb_a = np.asarray(spec["rgb_a"], dtype=np.float32) / 255.0
    rgb_b = np.asarray(spec["rgb_b"], dtype=np.float32) / 255.0

    count = min(len(frames), masks.shape[0])
    own_errors = []
    cross_leakages = []
    separations = []
    collision_ambiguity = []
    overlap_ratios = []
    for t in range(count):
        frame = frames[t]
        ma = _resize_mask(masks[t, 0], frame.shape[:2])
        mb = _resize_mask(masks[t, 1], frame.shape[:2])
        if ma.mean() < 1e-4 or mb.mean() < 1e-4:
            continue
        mean_a = _masked_mean_rgb(frame, ma)
        mean_b = _masked_mean_rgb(frame, mb)
        da_own = _rgb_distance(mean_a, rgb_a)
        da_other = _rgb_distance(mean_a, rgb_b)
        db_own = _rgb_distance(mean_b, rgb_b)
        db_other = _rgb_distance(mean_b, rgb_a)
        own_errors.extend([da_own, db_own])
        cross_leakages.extend(
            [
                da_own / (da_own + da_other + 1e-8),
                db_own / (db_own + db_other + 1e-8),
            ]
        )
        sep = _rgb_distance(mean_a, mean_b)
        separations.append(sep)
        overlap = float(np.minimum(ma, mb).sum() / (np.maximum(ma, mb).sum() + 1e-8))
        overlap_ratios.append(overlap)
        if overlap > 0.02:
            # Low A/B separation inside planned contact frames is treated as chimera ambiguity.
            collision_ambiguity.append(1.0 - min(1.0, sep / 0.45))

    mean_own = float(np.mean(own_errors)) if own_errors else 1.0
    mean_leak = float(np.mean(cross_leakages)) if cross_leakages else 1.0
    mean_sep = float(np.mean(separations)) if separations else 0.0
    mean_amb = float(np.mean(collision_ambiguity)) if collision_ambiguity else 1.0
    chimera_score = float(mean_own + mean_leak + mean_amb + (1.0 - min(1.0, mean_sep / 0.45)))
    return {
        "sample": str(sample_grid),
        "entity_own_color_error": mean_own,
        "cross_color_leakage": mean_leak,
        "pair_color_separation": mean_sep,
        "collision_ambiguity": mean_amb,
        "planned_overlap_ratio": float(np.mean(overlap_ratios)) if overlap_ratios else 0.0,
        "pair_chimera_score": chimera_score,
    }


def _make_visual(sample_dir: Path, output: Path, metrics: dict, columns: int, label_height: int) -> None:
    image_paths = [
        ("planned_sidecar", sample_dir / "planned_sidecar_grid.png"),
        ("no_sidecar", sample_dir / "no_sidecar_sample_grid.png"),
        ("full_sidecar", sample_dir / "full_sidecar_sample_grid.png"),
    ]
    loaded = [(name, Image.open(path).convert("RGB")) for name, path in image_paths if path.exists()]
    if not loaded:
        return
    thumb_w = 520
    padding = 10
    label_h = 88
    font = ImageDraw.ImageDraw(Image.new("RGB", (1, 1))).getfont()
    resized = []
    for name, im in loaded:
        ratio = thumb_w / im.width
        resized.append((name, im.resize((thumb_w, max(1, int(im.height * ratio))))))
    w = padding + sum(im.width + padding for _, im in resized)
    h = label_h + max(im.height for _, im in resized)
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    no = metrics.get("no_sidecar", {})
    full = metrics.get("full_sidecar", {})
    line1 = (
        f"pair_chimera_score lower better: no {no.get('pair_chimera_score', 0):.4f} "
        f"-> full {full.get('pair_chimera_score', 0):.4f}"
    )
    line2 = (
        f"leakage no {no.get('cross_color_leakage', 0):.4f} -> full {full.get('cross_color_leakage', 0):.4f}; "
        f"separation no {no.get('pair_color_separation', 0):.4f} -> full {full.get('pair_color_separation', 0):.4f}"
    )
    draw.text((padding, 8), line1, fill="black", font=font)
    draw.text((padding, 30), line2, fill="black", font=font)
    x = padding
    for name, im in resized:
        draw.text((x, 58), name, fill="black", font=font)
        canvas.paste(im, (x, label_h))
        x += im.width + padding
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pairwise collision chimera from planned sidecars.")
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--visual-output", type=Path, default=None)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--label-height", type=int, default=26)
    args = parser.parse_args()

    metadata = json.loads((args.sample_dir / "collision_sample_metadata.json").read_text())
    spec = metadata["spec"]
    planned_path = args.sample_dir / "planned_sidecar.pt"
    results = {}
    for variant in ["no_sidecar", "full_sidecar", "latent_control", "prompt_bias"]:
        path = args.sample_dir / f"{variant}_sample_grid.png"
        if path.exists():
            results[variant] = _score_variant(path, planned_path, spec, args.columns, args.label_height)
    if "no_sidecar" in results and "full_sidecar" in results:
        no = results["no_sidecar"]
        full = results["full_sidecar"]
        results["delta_full_minus_no"] = {
            key: full[key] - no[key]
            for key in [
                "entity_own_color_error",
                "cross_color_leakage",
                "pair_color_separation",
                "collision_ambiguity",
                "pair_chimera_score",
            ]
        }
        results["gate"] = {
            "chimera_score_improved": results["delta_full_minus_no"]["pair_chimera_score"] < 0,
            "leakage_improved": results["delta_full_minus_no"]["cross_color_leakage"] < 0,
            "separation_improved": results["delta_full_minus_no"]["pair_color_separation"] > 0,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.visual_output is not None:
        _make_visual(args.sample_dir, args.visual_output, results, args.columns, args.label_height)
    print(json.dumps(results.get("gate", {}), indent=2))


if __name__ == "__main__":
    main()
