from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "Databoost/VidData"
DEFAULT_DOWNLOAD_DIR = Path("/home/dongwoo39/datasets/viddata_raw")
DEFAULT_OUTPUT_DIR = Path("data/viddata")
CSV_RELATIVE_PATH = Path("data/train/VidData.csv")
VIDEO_DIR = Path("video")


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _maybe_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _resolve_video_path(raw_root: Path, video_name: str) -> Path:
    path = raw_root / VIDEO_DIR / video_name
    if path.exists():
        return path
    candidates = list((raw_root / VIDEO_DIR).rglob(video_name))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find video file for {video_name!r} under {raw_root / VIDEO_DIR}")


def _normalize_caption(row: Dict[str, str]) -> str:
    caption = row.get("Caption") or row.get("caption") or ""
    return str(caption).strip()


def _build_manifest_row(raw_root: Path, row: Dict[str, str]) -> Dict[str, str]:
    video_name = str(row.get("video_name") or row.get("video") or "").strip()
    if not video_name:
        raise ValueError("Missing video_name column in VidData CSV row")
    video_path = _resolve_video_path(raw_root, video_name)
    return {
        "video_path": str(video_path),
        "caption": _normalize_caption(row),
        "video_name": video_name,
        "temporal_consistency_score": str(row.get("temporal_consistency_score", "")).strip(),
        "motion_score": str(row.get("motion_score", "")).strip(),
        "fps": str(row.get("fps", "")).strip(),
        "frames": str(row.get("frames", "")).strip(),
        "duration_seconds": str(row.get("duration_seconds", "")).strip(),
        "camera_motion": str(row.get("camera_motion", "")).strip(),
        "source_repo": DEFAULT_REPO_ID,
    }


def _filter_rows(
    rows: Sequence[Dict[str, str]],
    min_motion_score: float | None,
    min_temporal_consistency: float | None,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        motion = _maybe_float(row.get("motion_score"))
        consistency = _maybe_float(row.get("temporal_consistency_score"))
        if min_motion_score is not None and (motion is None or motion < min_motion_score):
            continue
        if min_temporal_consistency is not None and (consistency is None or consistency < min_temporal_consistency):
            continue
        out.append(row)
    return out


def _split_rows(
    rows: Sequence[Dict[str, str]],
    val_ratio: float,
    seed: int,
    no_shuffle: bool,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rows = list(rows)
    if not no_shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    val_count = max(1, int(round(len(rows) * val_ratio))) if rows else 0
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]
    if not train_rows and val_rows:
        train_rows, val_rows = val_rows, []
    return train_rows, val_rows


def _write_jsonl(rows: Iterable[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(rows: Sequence[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _download_dataset(repo_id: str, download_dir: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(download_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["README.md", str(CSV_RELATIVE_PATH), "video/*.mp4"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare Databoost/VidData manifests.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--download-dir", type=str, default=str(DEFAULT_DOWNLOAD_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-motion-score", type=float, default=None)
    parser.add_argument("--min-temporal-consistency", type=float, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        _download_dataset(args.repo_id, download_dir)

    csv_path = download_dir / CSV_RELATIVE_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing VidData CSV: {csv_path}")

    rows = _load_rows(csv_path)
    filtered_rows = _filter_rows(rows, args.min_motion_score, args.min_temporal_consistency)
    if args.max_rows is not None:
        filtered_rows = filtered_rows[: args.max_rows]

    train_rows_raw, val_rows_raw = _split_rows(filtered_rows, args.val_ratio, args.seed, args.no_shuffle)
    raw_root = download_dir
    train_rows = [_build_manifest_row(raw_root, row) for row in train_rows_raw]
    val_rows = [_build_manifest_row(raw_root, row) for row in val_rows_raw]

    train_jsonl = output_dir / "train.jsonl"
    val_jsonl = output_dir / "val.jsonl"
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    meta_json = output_dir / "meta.json"

    _write_jsonl(train_rows, train_jsonl)
    _write_jsonl(val_rows, val_jsonl)
    _write_csv(train_rows, train_csv)
    _write_csv(val_rows, val_csv)

    motion_scores = [_maybe_float(row.get("motion_score")) for row in filtered_rows]
    motion_scores = [score for score in motion_scores if score is not None]
    consistency_scores = [_maybe_float(row.get("temporal_consistency_score")) for row in filtered_rows]
    consistency_scores = [score for score in consistency_scores if score is not None]

    meta = {
        "repo_id": args.repo_id,
        "download_dir": str(download_dir),
        "output_dir": str(output_dir),
        "source_csv": str(csv_path),
        "total_rows": len(rows),
        "filtered_rows": len(filtered_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "filters": {
            "min_motion_score": args.min_motion_score,
            "min_temporal_consistency": args.min_temporal_consistency,
        },
        "motion_score_min": min(motion_scores) if motion_scores else None,
        "motion_score_max": max(motion_scores) if motion_scores else None,
        "temporal_consistency_min": min(consistency_scores) if consistency_scores else None,
        "temporal_consistency_max": max(consistency_scores) if consistency_scores else None,
    }
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Prepared VidData from {csv_path}")
    print(f"  total_rows: {len(rows)}")
    print(f"  filtered_rows: {len(filtered_rows)}")
    print(f"  train_rows: {len(train_rows)} -> {train_jsonl}")
    print(f"  val_rows: {len(val_rows)} -> {val_jsonl}")
    print(f"  meta: {meta_json}")
    if train_rows:
        preview = train_rows[0]
        print(f"  preview_video: {preview['video_name']}")
        print(f"  preview_caption: {preview['caption']}")
        print(f"  preview_path: {preview['video_path']}")


if __name__ == "__main__":
    main()
