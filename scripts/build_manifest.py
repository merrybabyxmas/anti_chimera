from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.gif', '.webm'}


def load_caption_table(path: Path) -> Dict[str, str]:
    if path.suffix == '.jsonl':
        table = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                table[row['video']] = row['caption']
        return table
    if path.suffix == '.csv':
        table = {}
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                table[row['video']] = row['caption']
        return table
    raise ValueError('caption table must be .csv or .jsonl')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-root', type=str, required=True)
    parser.add_argument('--captions', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--sidecar-root', type=str, default=None)
    args = parser.parse_args()

    video_root = Path(args.video_root)
    caption_table = load_caption_table(Path(args.captions))
    sidecar_root = Path(args.sidecar_root) if args.sidecar_root else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        for path in sorted(video_root.rglob('*')):
            if path.suffix.lower() not in VIDEO_EXTS:
                continue
            rel = path.relative_to(video_root).as_posix()
            caption = caption_table.get(rel) or caption_table.get(path.name)
            if caption is None:
                continue
            row = {
                'video_path': str(path),
                'caption': caption,
            }
            if sidecar_root is not None:
                stem = Path(rel).with_suffix('')
                tracks = sidecar_root / f'{stem}_tracks.npy'
                depth = sidecar_root / f'{stem}_depth.npy'
                visibility = sidecar_root / f'{stem}_visibility.npy'
                if tracks.exists():
                    row['tracks_path'] = str(tracks)
                if depth.exists():
                    row['depth_path'] = str(depth)
                if visibility.exists():
                    row['visibility_path'] = str(visibility)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
