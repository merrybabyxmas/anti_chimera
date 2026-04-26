from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

DEFAULT_INPUT_DIR = Path('data/viddata')
DEFAULT_OUTPUT_DIR = Path('data/viddata_sidecar')
DEFAULT_MAX_OBJECTS = 4


def _load_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: Sequence[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _sample_indices(total_frames: int, num_frames: int) -> np.ndarray:
    total_frames = max(1, int(total_frames))
    num_frames = max(1, int(num_frames))
    if total_frames == 1:
        return np.zeros(num_frames, dtype=np.int64)
    return np.linspace(0, total_frames - 1, num_frames).round().astype(np.int64)


def _load_frames(video_path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f'failed to read any frames from {video_path}')
    return frames


def _resize_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)


def _build_motion_map(frames: np.ndarray) -> np.ndarray:
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames], axis=0).astype(np.float32) / 255.0
    motion = np.zeros_like(gray, dtype=np.float32)
    for t in range(1, gray.shape[0]):
        diff = cv2.absdiff(gray[t], gray[t - 1]).astype(np.float32)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        motion[t] = diff
    motion -= motion.min()
    motion /= motion.max() + 1e-6
    return motion


def _build_tracks(motion: np.ndarray, image_size: int, max_objects: int) -> tuple[np.ndarray, np.ndarray]:
    T = motion.shape[0]
    tracks = np.zeros((T, max_objects, 4), dtype=np.float32)
    visibility = np.zeros((T, max_objects), dtype=np.float32)
    min_area = max(16, int(image_size * image_size * 0.002))

    for t in range(T):
        mask = motion[t]
        threshold = max(float(mask.mean() + mask.std() * 0.75), float(np.percentile(mask, 85)), 0.12)
        binary = (mask >= threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        components: List[tuple[int, int, int, int, int]] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            components.append((area, x, y, x + w, y + h))
        if not components and binary.any():
            ys, xs = np.where(binary > 0)
            if len(xs) > 0 and len(ys) > 0:
                components.append((int(len(xs)), int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)))
        components.sort(key=lambda item: item[0], reverse=True)
        for k, (_, x1, y1, x2, y2) in enumerate(components[:max_objects]):
            tracks[t, k] = np.array([x1 / image_size, y1 / image_size, x2 / image_size, y2 / image_size], dtype=np.float32)
            visibility[t, k] = 1.0
    return tracks, visibility


def _build_sidecars(video_path: Path, num_frames: int, image_size: int, max_objects: int) -> Dict[str, np.ndarray]:
    raw_frames = _load_frames(video_path)
    idxs = _sample_indices(len(raw_frames), num_frames)
    sampled = np.stack([_resize_frame(raw_frames[i], image_size) for i in idxs], axis=0)
    motion = _build_motion_map(sampled)
    depth = (1.0 - motion).astype(np.float32)
    tracks, visibility = _build_tracks(motion, image_size=image_size, max_objects=max_objects)
    return {
        'tracks': tracks,
        'depth': depth,
        'visibility': visibility,
    }


def _prepare_manifest(
    manifest_path: Path,
    output_manifest_path: Path,
    sidecar_dir: Path,
    num_frames: int,
    image_size: int,
    max_objects: int,
    overwrite: bool,
) -> dict:
    rows = _load_jsonl(manifest_path)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    enriched_rows: List[Dict[str, str]] = []
    motion_stats: List[float] = []

    for idx, row in enumerate(rows):
        video_path = Path(row['video_path'])
        if not video_path.exists():
            raise FileNotFoundError(f'missing video: {video_path}')
        sidecar_path = sidecar_dir / f'{video_path.stem}.npz'
        if overwrite or not sidecar_path.exists():
            sidecars = _build_sidecars(video_path, num_frames=num_frames, image_size=image_size, max_objects=max_objects)
            np.savez_compressed(sidecar_path, **sidecars)
        else:
            loaded = np.load(sidecar_path)
            sidecars = {key: loaded[key] for key in loaded.files}
        enriched = dict(row)
        enriched['tracks_path'] = str(sidecar_path)
        enriched['depth_path'] = str(sidecar_path)
        enriched['visibility_path'] = str(sidecar_path)
        enriched_rows.append(enriched)
        motion_stats.append(float(np.asarray(sidecars['depth']).mean()))

    _write_jsonl(enriched_rows, output_manifest_path)
    meta = {
        'input_manifest': str(manifest_path),
        'output_manifest': str(output_manifest_path),
        'sidecar_dir': str(sidecar_dir),
        'num_rows': len(enriched_rows),
        'num_frames': num_frames,
        'image_size': image_size,
        'max_objects': max_objects,
        'mean_depth': float(np.mean(motion_stats)) if motion_stats else None,
        'min_depth': float(np.min(motion_stats)) if motion_stats else None,
        'max_depth': float(np.max(motion_stats)) if motion_stats else None,
    }
    (output_manifest_path.parent / 'meta_sidecar.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare VidData sidecar manifests from real videos.')
    parser.add_argument('--input-dir', type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--train-manifest', type=str, default='train.jsonl')
    parser.add_argument('--val-manifest', type=str, default='val.jsonl')
    parser.add_argument('--num-frames', type=int, default=25)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--max-objects', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    train_manifest = input_dir / args.train_manifest
    val_manifest = input_dir / args.val_manifest
    if not train_manifest.exists():
        raise FileNotFoundError(f'missing train manifest: {train_manifest}')
    if not val_manifest.exists():
        raise FileNotFoundError(f'missing val manifest: {val_manifest}')

    train_output = output_dir / 'train.jsonl'
    val_output = output_dir / 'val.jsonl'
    train_sidecar_dir = output_dir / 'sidecars' / 'train'
    val_sidecar_dir = output_dir / 'sidecars' / 'val'

    train_meta = _prepare_manifest(
        train_manifest,
        train_output,
        train_sidecar_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        max_objects=args.max_objects,
        overwrite=args.overwrite,
    )
    val_meta = _prepare_manifest(
        val_manifest,
        val_output,
        val_sidecar_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        max_objects=args.max_objects,
        overwrite=args.overwrite,
    )

    combined_meta = {
        'train': train_meta,
        'val': val_meta,
    }
    (output_dir / 'meta.json').write_text(json.dumps(combined_meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(combined_meta, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
