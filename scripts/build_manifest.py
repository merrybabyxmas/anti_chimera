from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.gif', '.webm'}


def load_caption_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if path.suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(row)
        return rows
    if path.suffix == '.csv':
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows
    raise ValueError('caption table must be .csv or .jsonl')


def build_caption_table(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    table: Dict[str, Dict[str, str]] = {}
    for row in rows:
        video = row['video']
        table[video] = row
    return table


def maybe_add_sidecars(row: Dict[str, str], rel: str, sidecar_root: Path | None) -> None:
    if sidecar_root is None:
        return
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


def collect_manifest_rows(video_root: Path, caption_rows: Sequence[Dict[str, str]], sidecar_root: Path | None, strict: bool) -> List[Dict[str, str]]:
    caption_table = build_caption_table(caption_rows)
    manifest_rows: List[Dict[str, str]] = []
    unmatched_videos: List[str] = []
    for path in sorted(video_root.rglob('*')):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        rel = path.relative_to(video_root).as_posix()
        matched = caption_table.get(rel) or caption_table.get(path.name)
        if matched is None:
            unmatched_videos.append(rel)
            continue
        row = {
            'video_path': str(path),
            'caption': matched['caption'],
        }
        if 'entities' in matched and matched['entities']:
            row['entities'] = matched['entities']
        maybe_add_sidecars(row, rel, sidecar_root)
        manifest_rows.append(row)

    if strict and unmatched_videos:
        preview = '\n'.join(unmatched_videos[:10])
        raise ValueError(f'Found videos without captions. First unmatched entries:\n{preview}')
    return manifest_rows


def write_jsonl(rows: Sequence[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def split_rows(rows: List[Dict[str, str]], val_ratio: float, seed: int) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(round(len(rows) * val_ratio))) if rows else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    if not train_rows and val_rows:
        train_rows, val_rows = val_rows, []
    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-root', type=str, required=True)
    parser.add_argument('--captions', type=str, required=True)
    parser.add_argument('--out', type=str, default=None, help='Single output manifest path.')
    parser.add_argument('--train-out', type=str, default=None, help='Train manifest output path.')
    parser.add_argument('--val-out', type=str, default=None, help='Validation manifest output path.')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sidecar-root', type=str, default=None)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()

    if args.out is None and args.train_out is None:
        raise ValueError('Provide either --out or --train-out.')

    video_root = Path(args.video_root)
    caption_rows = load_caption_rows(Path(args.captions))
    sidecar_root = Path(args.sidecar_root) if args.sidecar_root else None
    rows = collect_manifest_rows(video_root, caption_rows, sidecar_root, strict=args.strict)

    if args.out is not None:
        write_jsonl(rows, Path(args.out))
        print(f'Wrote {len(rows)} rows to {args.out}')
        return

    train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.seed)
    write_jsonl(train_rows, Path(args.train_out))
    if args.val_out is not None:
        write_jsonl(val_rows, Path(args.val_out))
    print(f'Wrote {len(train_rows)} train rows to {args.train_out}')
    if args.val_out is not None:
        print(f'Wrote {len(val_rows)} val rows to {args.val_out}')


if __name__ == '__main__':
    main()
