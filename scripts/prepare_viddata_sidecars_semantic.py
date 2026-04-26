from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anti_chimera.text import PromptParser

DEFAULT_INPUT_DIR = Path('data/viddata')
DEFAULT_OUTPUT_DIR = Path('data/viddata_sidecar_semantic')
DEFAULT_MAX_OBJECTS = 4
DEFAULT_LIMIT = 0
DEFAULT_MODEL_NAME = 'facebook/detr-resnet-50'

COCO_ALIASES = {
    'person': {'person', 'man', 'woman', 'boy', 'girl', 'people'},
    'bicycle': {'bicycle', 'bike', 'cycle'},
    'car': {'car', 'auto', 'vehicle', 'truck', 'van', 'bus'},
    'motorcycle': {'motorcycle', 'motorbike'},
    'airplane': {'airplane', 'plane'},
    'train': {'train', 'tram'},
    'boat': {'boat', 'ship', 'ferry'},
    'bird': {'bird'},
    'cat': {'cat'},
    'dog': {'dog'},
    'horse': {'horse'},
    'sheep': {'sheep'},
    'cow': {'cow'},
    'elephant': {'elephant'},
    'bear': {'bear'},
    'zebra': {'zebra'},
    'giraffe': {'giraffe'},
    'chair': {'chair', 'seat'},
    'couch': {'couch', 'sofa'},
    'potted plant': {'plant', 'potted plant', 'tree'},
    'bed': {'bed'},
    'dining table': {'table', 'dining table'},
    'tv': {'tv', 'monitor', 'screen'},
    'laptop': {'laptop', 'computer', 'keyboard'},
    'cell phone': {'phone', 'cell phone', 'mobile'},
}


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


def _load_sampled_frames(video_path: Path, num_frames: int, image_size: int) -> np.ndarray:
    raw_frames = _load_frames(video_path)
    idxs = _sample_indices(len(raw_frames), num_frames)
    return np.stack([_resize_frame(raw_frames[i], image_size) for i in idxs], axis=0)


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


def _build_motion_sidecar(frames: np.ndarray, image_size: int, max_objects: int) -> Dict[str, np.ndarray]:
    motion = _build_motion_map(frames)
    depth = (1.0 - motion).astype(np.float32)
    T = motion.shape[0]
    tracks = np.zeros((T, max_objects, 4), dtype=np.float32)
    visibility = np.zeros((T, max_objects), dtype=np.float32)
    min_area = max(16, int(image_size * image_size * 0.002))
    for t in range(T):
        mask = motion[t]
        threshold = max(float(mask.mean() + mask.std() * 0.75), float(np.percentile(mask, 85)), 0.12)
        binary = (mask >= threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        components: List[Tuple[int, int, int, int, int]] = []
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
    return {'tracks': tracks, 'depth': depth, 'visibility': visibility}


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(area_a + area_b - inter, 1e-6)
    return float(inter / union)


def _aliases_from_entities(entities: Sequence[str], caption: str) -> set[str]:
    parser = PromptParser()
    parsed = parser.parse(caption)
    tokens = set(parsed.entities or [])
    for ent in entities:
        tokens.add(ent.lower())
    labels: set[str] = set()
    for token in tokens:
        for label, aliases in COCO_ALIASES.items():
            if token == label or any(alias in token for alias in aliases):
                labels.add(label)
    return labels


def _load_detector(device: torch.device, model_name: str = DEFAULT_MODEL_NAME):
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    categories = dict(model.config.id2label)
    return processor, model, categories


def _build_semantic_sidecar(
    frames: np.ndarray,
    caption: str,
    entities: Sequence[str],
    image_size: int,
    max_objects: int,
    processor,
    detector,
    categories: Dict[int, str],
    device: torch.device,
    score_thresh: float = 0.25,
) -> Dict[str, np.ndarray]:
    T = frames.shape[0]
    tracks = np.zeros((T, max_objects, 4), dtype=np.float32)
    visibility = np.zeros((T, max_objects), dtype=np.float32)
    depth = np.ones((T, image_size, image_size), dtype=np.float32)
    prev_boxes: list[np.ndarray | None] = [None] * max_objects
    prev_labels: list[str | None] = [None] * max_objects
    wanted_labels = _aliases_from_entities(entities, caption)
    any_detection = False

    rgbs = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    target_sizes = torch.tensor([[frame.shape[0], frame.shape[1]] for frame in frames], device=device)

    with torch.inference_mode():
        inputs = processor(images=rgbs, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = detector(**inputs)
        batch_results = processor.post_process_object_detection(outputs, threshold=score_thresh, target_sizes=target_sizes)

    for t, results in enumerate(batch_results):
        boxes = results['boxes'].detach().cpu().numpy() if len(results['boxes']) else np.zeros((0, 4), dtype=np.float32)
        labels = results['labels'].detach().cpu().numpy() if len(results['labels']) else np.zeros((0,), dtype=np.int64)
        scores = results['scores'].detach().cpu().numpy() if len(results['scores']) else np.zeros((0,), dtype=np.float32)

        dets: List[tuple[float, str, np.ndarray]] = []
        for box, label_id, score in zip(boxes, labels, scores):
            label_name = categories.get(int(label_id), str(int(label_id)))
            if wanted_labels and label_name not in wanted_labels:
                continue
            dets.append((float(score), label_name, box.astype(np.float32)))
        if not dets:
            for box, label_id, score in zip(boxes, labels, scores):
                label_name = categories.get(int(label_id), str(int(label_id)))
                dets.append((float(score), label_name, box.astype(np.float32)))
        dets.sort(key=lambda item: item[0], reverse=True)
        dets = dets[:max_objects]
        if dets:
            any_detection = True

        assigned: set[int] = set()
        next_boxes: list[np.ndarray | None] = [None] * max_objects
        next_labels: list[str | None] = [None] * max_objects
        foreground = np.zeros((image_size, image_size), dtype=np.float32)

        for score, label_name, box in dets:
            best_slot = None
            best_iou = 0.0
            for slot in range(max_objects):
                if slot in assigned:
                    continue
                prev_box = prev_boxes[slot]
                if prev_box is None:
                    continue
                if prev_labels[slot] is not None and prev_labels[slot] != label_name:
                    continue
                iou = _box_iou(prev_box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_slot = slot
            if best_slot is None or best_iou < 0.15:
                for slot in range(max_objects):
                    if slot not in assigned and prev_boxes[slot] is None:
                        best_slot = slot
                        break
            if best_slot is None:
                continue
            assigned.add(best_slot)
            next_boxes[best_slot] = box
            next_labels[best_slot] = label_name
            x1, y1, x2, y2 = box
            tracks[t, best_slot] = np.array([x1 / image_size, y1 / image_size, x2 / image_size, y2 / image_size], dtype=np.float32)
            visibility[t, best_slot] = score
            x1i, y1i = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
            x2i, y2i = min(image_size, int(np.ceil(x2))), min(image_size, int(np.ceil(y2)))
            foreground[y1i:y2i, x1i:x2i] = np.maximum(foreground[y1i:y2i, x1i:x2i], score)

        if dets:
            depth[t] = 1.0 - np.clip(foreground, 0.0, 1.0)
        prev_boxes = next_boxes
        prev_labels = next_labels

    if not any_detection:
        return _build_motion_sidecar(frames, image_size=image_size, max_objects=max_objects)
    return {'tracks': tracks, 'depth': depth.astype(np.float32), 'visibility': visibility}


def _prepare_manifest(
    manifest_path: Path,
    output_manifest_path: Path,
    sidecar_dir: Path,
    num_frames: int,
    image_size: int,
    max_objects: int,
    overwrite: bool,
    processor,
    detector,
    categories: Dict[int, str],
    device: torch.device,
    limit: int = 0,
) -> dict:
    rows = _load_jsonl(manifest_path)
    if limit > 0:
        rows = rows[:limit]
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    enriched_rows: List[Dict[str, str]] = []
    score_stats: List[float] = []

    for idx, row in enumerate(rows):
        if idx == 0 or (idx + 1) % 10 == 0:
            print(f"processing {idx + 1}/{len(rows)}: {row.get('video_path', '')}", flush=True)
        video_path = Path(row['video_path'])
        if not video_path.exists():
            raise FileNotFoundError(f'missing video: {video_path}')
        sidecar_path = sidecar_dir / f'{video_path.stem}.npz'
        if overwrite or not sidecar_path.exists():
            frames = _load_sampled_frames(video_path, num_frames=num_frames, image_size=image_size)
            sidecars = _build_semantic_sidecar(
                frames,
                caption=row.get('caption', ''),
                entities=row.get('entities', []),
                image_size=image_size,
                max_objects=max_objects,
                processor=processor,
                detector=detector,
                categories=categories,
                device=device,
            )
            np.savez_compressed(sidecar_path, **sidecars)
        else:
            loaded = np.load(sidecar_path)
            sidecars = {key: loaded[key] for key in loaded.files}
        enriched = dict(row)
        enriched['tracks_path'] = str(sidecar_path)
        enriched['depth_path'] = str(sidecar_path)
        enriched['visibility_path'] = str(sidecar_path)
        enriched_rows.append(enriched)
        score_stats.append(float(np.asarray(sidecars['visibility']).mean()))

    _write_jsonl(enriched_rows, output_manifest_path)
    meta = {
        'input_manifest': str(manifest_path),
        'output_manifest': str(output_manifest_path),
        'sidecar_dir': str(sidecar_dir),
        'num_rows': len(enriched_rows),
        'num_frames': num_frames,
        'image_size': image_size,
        'max_objects': max_objects,
        'mean_visibility': float(np.mean(score_stats)) if score_stats else None,
        'min_visibility': float(np.min(score_stats)) if score_stats else None,
        'max_visibility': float(np.max(score_stats)) if score_stats else None,
    }
    (output_manifest_path.parent / 'meta_sidecar.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare VidData semantic sidecar manifests from real videos.')
    parser.add_argument('--input-dir', type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--train-manifest', type=str, default='train.jsonl')
    parser.add_argument('--val-manifest', type=str, default='val.jsonl')
    parser.add_argument('--num-frames', type=int, default=25)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--max-objects', type=int, default=4)
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, help='Optional row limit for smoke subsets.')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model-name', type=str, default=DEFAULT_MODEL_NAME)
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

    device = torch.device(args.device)
    processor, detector, categories = _load_detector(device, model_name=args.model_name)

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
        processor=processor,
        detector=detector,
        categories=categories,
        device=device,
        limit=args.limit,
    )
    val_meta = _prepare_manifest(
        val_manifest,
        val_output,
        val_sidecar_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        max_objects=args.max_objects,
        overwrite=args.overwrite,
        processor=processor,
        detector=detector,
        categories=categories,
        device=device,
        limit=args.limit,
    )

    combined_meta = {'train': train_meta, 'val': val_meta}
    (output_dir / 'meta.json').write_text(json.dumps(combined_meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(combined_meta, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
