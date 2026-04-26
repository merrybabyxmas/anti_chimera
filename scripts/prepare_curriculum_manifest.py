from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np


SUBJECT_WORDS = {
    "person",
    "people",
    "man",
    "men",
    "woman",
    "women",
    "boy",
    "girl",
    "driver",
    "passenger",
    "child",
    "dog",
    "cat",
    "animal",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False), file=f)


def _as_float(row: dict, key: str, default: float) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _subject_score(caption: str) -> float:
    tokens = re.findall(r"[a-z]+", caption.lower())
    hits = sum(1 for token in tokens if token in SUBJECT_WORDS)
    return min(1.0, hits / 3.0)


def _sidecar_score(row: dict, root: Path, scan_sidecars: bool) -> float:
    if not scan_sidecars:
        return 0.5
    sidecar = row.get("tracks_path") or row.get("depth_path") or row.get("masks_path")
    if not sidecar:
        return 0.0
    path = Path(sidecar)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return 0.0
    try:
        with np.load(path) as z:
            visibility = float(np.asarray(z.get("visibility", 0.0)).mean())
            masks = float(np.asarray(z.get("masks", 0.0)).mean())
            depth = float(np.asarray(z.get("depth", 0.0)).std())
            tracks = np.asarray(z.get("tracks", 0.0))
            track_signal = float((tracks[..., 2] > tracks[..., 0]).mean()) if tracks.ndim == 3 else 0.0
    except Exception:
        return 0.0
    return max(0.0, min(1.0, 0.35 * visibility + 0.25 * min(1.0, masks * 4.0) + 0.2 * min(1.0, depth * 4.0) + 0.2 * track_signal))


def _score_row(row: dict, root: Path, scan_sidecars: bool) -> float:
    temporal = _as_float(row, "temporal_consistency_score", 0.5)
    motion = _as_float(row, "motion_score", 0.5)
    camera_motion = _as_float(row, "camera_motion", 0.5)
    caption = str(row.get("caption", ""))
    motion_band = 1.0 - min(1.0, abs(motion - 0.55) / 0.55)
    camera_stability = 1.0 - min(1.0, camera_motion)
    return (
        0.30 * temporal
        + 0.20 * motion_band
        + 0.20 * camera_stability
        + 0.15 * _subject_score(caption)
        + 0.15 * _sidecar_score(row, root, scan_sidecars)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 32, 256])
    parser.add_argument("--no-scan-sidecars", action="store_true")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir)
    scan_sidecars = not args.no_scan_sidecars
    train_rows = _read_jsonl(Path(args.train_manifest))
    val_rows = _read_jsonl(Path(args.val_manifest))

    train_ranked = sorted(train_rows, key=lambda row: _score_row(row, root, scan_sidecars), reverse=True)
    val_ranked = sorted(val_rows, key=lambda row: _score_row(row, root, scan_sidecars), reverse=True)

    for size in args.sizes:
        train_out = train_ranked[: min(size, len(train_ranked))]
        val_size = min(max(4, size // 8), len(val_ranked))
        val_out = val_ranked[:val_size]
        _write_jsonl(output_dir / f"curated{size}_train.jsonl", train_out)
        _write_jsonl(output_dir / f"curated{size}_val.jsonl", val_out)
        print(f"curated{size}: train={len(train_out)} val={len(val_out)}")


if __name__ == "__main__":
    main()
