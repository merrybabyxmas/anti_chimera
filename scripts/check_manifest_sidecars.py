from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _resolve(root_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else root_dir / p


def _video_frame_count(path: Path) -> int:
    reader = imageio.get_reader(path)
    count = 0
    try:
        while True:
            try:
                _ = reader.get_next_data()
            except (IndexError, StopIteration):
                break
            count += 1
    finally:
        reader.close()
    return count


def _load_array(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    if path.suffix == '.npy':
        return np.load(path)
    if path.suffix == '.npz':
        loaded = np.load(path)
        return loaded[loaded.files[0]]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=str)
    parser.add_argument('--root-dir', default='.', type=str)
    parser.add_argument('--num-frames', required=True, type=int)
    parser.add_argument('--image-size', required=True, type=int)
    parser.add_argument('--max-objects', required=True, type=int)
    parser.add_argument('--limit', default=0, type=int)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    root_dir = Path(args.root_dir)
    items = _load_jsonl(manifest)
    if args.limit > 0:
        items = items[: args.limit]

    summary = {
        'items': len(items),
        'missing_video': 0,
        'video_read_failures': 0,
        'tracks_present': 0,
        'depth_present': 0,
        'visibility_present': 0,
        'masks_present': 0,
        'flow_present': 0,
        'occlusion_present': 0,
        'shape_warnings': 0,
    }

    for idx, item in enumerate(items):
        video_path = _resolve(root_dir, item.get('video_path'))
        if video_path is None or not video_path.exists():
            summary['missing_video'] += 1
            print(f'[missing_video] idx={idx} path={video_path}')
            continue
        try:
            frame_count = _video_frame_count(video_path)
            if frame_count <= 0:
                raise RuntimeError('no frames')
        except Exception as exc:
            summary['video_read_failures'] += 1
            print(f'[video_read_failure] idx={idx} path={video_path} err={exc}')
            continue

        specs = [
            ('tracks_path', 'tracks_present', (args.max_objects, 4)),
            ('depth_path', 'depth_present', None),
            ('visibility_path', 'visibility_present', (args.max_objects,)),
            ('masks_path', 'masks_present', (args.max_objects, args.image_size, args.image_size)),
            ('flow_path', 'flow_present', (2, args.image_size, args.image_size)),
            ('occlusion_path', 'occlusion_present', (args.image_size, args.image_size)),
        ]
        for key, counter, trailing_shape in specs:
            sidecar = _resolve(root_dir, item.get(key))
            arr = _load_array(sidecar)
            if arr is None:
                continue
            summary[counter] += 1
            if arr.shape[0] <= 0:
                summary['shape_warnings'] += 1
                print(f'[shape_warning] idx={idx} key={key} shape={arr.shape} reason=empty_time_axis')
                continue
            if trailing_shape is not None and tuple(arr.shape[1:]) != tuple(trailing_shape):
                summary['shape_warnings'] += 1
                print(f'[shape_warning] idx={idx} key={key} shape={arr.shape} expected=(*,{trailing_shape})')
            if arr.shape[0] < args.num_frames:
                summary['shape_warnings'] += 1
                print(f'[shape_warning] idx={idx} key={key} shape={arr.shape} reason=too_few_frames_for_requested_sampling')

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
