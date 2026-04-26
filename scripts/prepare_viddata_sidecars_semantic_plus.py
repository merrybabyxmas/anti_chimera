from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    DPTForDepthEstimation,
    DPTImageProcessor,
    DetrForObjectDetection,
    DetrImageProcessor,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

DEFAULT_INPUT_DIR = Path('data/viddata')
DEFAULT_OUTPUT_DIR = Path('data/viddata_sidecar_semantic_plus')
DEFAULT_MAX_OBJECTS = 4
DEFAULT_NUM_FRAMES = 25
DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 2
DEFAULT_DEVICE = 'cuda'


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


def _to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _load_detectors(device: torch.device):
    dtype = torch.float16 if device.type == 'cuda' else torch.float32

    detr_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', use_safetensors=True)
    detr_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', torch_dtype=dtype, use_safetensors=True)
    detr_model.to(device).eval()

    depth_processor = DPTImageProcessor.from_pretrained('Intel/dpt-hybrid-midas', use_safetensors=True)
    depth_model = DPTForDepthEstimation.from_pretrained('Intel/dpt-hybrid-midas', torch_dtype=dtype, use_safetensors=True)
    depth_model.to(device).eval()

    mask_processor = Mask2FormerImageProcessor.from_pretrained('facebook/mask2former-swin-base-coco-instance', use_safetensors=True)
    mask_model = Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-base-coco-instance', torch_dtype=dtype, use_safetensors=True)
    mask_model.to(device).eval()

    return {
        'detr_processor': detr_processor,
        'detr_model': detr_model,
        'depth_processor': depth_processor,
        'depth_model': depth_model,
        'mask_processor': mask_processor,
        'mask_model': mask_model,
    }




def _move_inputs(inputs: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    moved: Dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved
def _normalize_to_unit(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    denom = arr.max() - arr.min()
    if denom < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / denom


def _prepare_boxes(
    processor: DetrImageProcessor,
    model: DetrForObjectDetection,
    frame: np.ndarray,
    image_size: int,
    max_objects: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pil = _to_pil(frame)
    inputs = processor(images=pil, return_tensors='pt')
    inputs = _move_inputs(inputs, model)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = [pil.size[::-1]]
    results = processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
    boxes = results['boxes'].detach().cpu().numpy() if len(results['boxes']) else np.zeros((0, 4), dtype=np.float32)
    scores = results['scores'].detach().cpu().numpy() if len(results['scores']) else np.zeros((0,), dtype=np.float32)
    order = np.argsort(-scores)[:max_objects] if len(scores) else np.zeros((0,), dtype=np.int64)
    tracks = np.zeros((max_objects, 4), dtype=np.float32)
    visibility = np.zeros((max_objects,), dtype=np.float32)
    for out_idx, det_idx in enumerate(order):
        x1, y1, x2, y2 = boxes[det_idx].tolist()
        x1 = np.clip(x1 / image_size, 0.0, 1.0)
        y1 = np.clip(y1 / image_size, 0.0, 1.0)
        x2 = np.clip(x2 / image_size, 0.0, 1.0)
        y2 = np.clip(y2 / image_size, 0.0, 1.0)
        tracks[out_idx] = np.array([x1, y1, x2, y2], dtype=np.float32)
        visibility[out_idx] = float(scores[det_idx])
    return tracks, visibility


def _prepare_masks(
    processor: Mask2FormerImageProcessor,
    model: Mask2FormerForUniversalSegmentation,
    frame: np.ndarray,
    image_size: int,
    max_objects: int,
) -> np.ndarray:
    pil = _to_pil(frame)
    inputs = processor(images=pil, return_tensors='pt')
    inputs = _move_inputs(inputs, model)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = [pil.size[::-1]]
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.3,
        mask_threshold=0.5,
        overlap_mask_area_threshold=0.8,
        target_sizes=target_sizes,
        return_binary_maps=True,
    )[0]
    masks = results.get('segmentation')
    if masks is None:
        return np.zeros((max_objects, image_size, image_size), dtype=np.float32)
    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)
    if masks_np.ndim == 2:
        masks_np = masks_np[None]
    if masks_np.shape[0] > max_objects:
        masks_np = masks_np[:max_objects]
    out = np.zeros((max_objects, image_size, image_size), dtype=np.float32)
    for i in range(min(max_objects, masks_np.shape[0])):
        mask = masks_np[i].astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / (mask.max() + 1e-6)
        out[i] = mask
    return out


def _prepare_depth(
    processor: DPTImageProcessor,
    model: DPTForDepthEstimation,
    frame: np.ndarray,
    image_size: int,
) -> np.ndarray:
    pil = _to_pil(frame)
    inputs = processor(images=pil, return_tensors='pt')
    inputs = _move_inputs(inputs, model)
    with torch.no_grad():
        outputs = model(**inputs)
    depth_dict = processor.post_process_depth_estimation(outputs, target_sizes=[pil.size[::-1]])[0]
    depth = depth_dict['predicted_depth']
    if isinstance(depth, torch.Tensor):
        depth_np = depth.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth)
    if depth_np.ndim == 3:
        depth_np = depth_np[0]
    depth_np = cv2.resize(depth_np.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return _normalize_to_unit(depth_np)


def _compute_flow_and_occlusion(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames], axis=0).astype(np.float32) / 255.0
    T, H, W = gray.shape
    flow = np.zeros((T, 2, H, W), dtype=np.float32)
    occlusion = np.zeros((T, H, W), dtype=np.float32)
    for t in range(1, T):
        prev = gray[t - 1]
        curr = gray[t]
        fwd = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        bwd = cv2.calcOpticalFlowFarneback(curr, prev, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        flow[t] = np.transpose(fwd, (2, 0, 1)).astype(np.float32)
        mag = np.sqrt((fwd[..., 0] ** 2) + (fwd[..., 1] ** 2))
        consistency = np.sqrt(((fwd[..., 0] + bwd[..., 0]) ** 2) + ((fwd[..., 1] + bwd[..., 1]) ** 2))
        occlusion[t] = _normalize_to_unit(0.5 * mag + 0.5 * consistency)
    return flow, occlusion


def _build_sidecars(
    video_path: Path,
    num_frames: int,
    image_size: int,
    max_objects: int,
    detectors: Dict[str, object],
) -> Dict[str, np.ndarray]:
    raw_frames = _load_frames(video_path)
    idxs = _sample_indices(len(raw_frames), num_frames)
    sampled = np.stack([_resize_frame(raw_frames[i], image_size) for i in idxs], axis=0)

    tracks = np.zeros((num_frames, max_objects, 4), dtype=np.float32)
    visibility = np.zeros((num_frames, max_objects), dtype=np.float32)
    masks = np.zeros((num_frames, max_objects, image_size, image_size), dtype=np.float32)
    depth = np.zeros((num_frames, image_size, image_size), dtype=np.float32)

    for t, frame in enumerate(sampled):
        tracks[t], visibility[t] = _prepare_boxes(
            detectors['detr_processor'],
            detectors['detr_model'],
            frame,
            image_size=image_size,
            max_objects=max_objects,
        )
        masks[t] = _prepare_masks(
            detectors['mask_processor'],
            detectors['mask_model'],
            frame,
            image_size=image_size,
            max_objects=max_objects,
        )
        depth[t] = _prepare_depth(
            detectors['depth_processor'],
            detectors['depth_model'],
            frame,
            image_size=image_size,
        )

    flow, occlusion = _compute_flow_and_occlusion(sampled)
    return {
        'tracks': tracks,
        'masks': masks,
        'depth': depth,
        'visibility': visibility,
        'flow': flow,
        'occlusion': occlusion,
    }


def _prepare_manifest(
    manifest_path: Path,
    output_manifest_path: Path,
    sidecar_dir: Path,
    num_frames: int,
    image_size: int,
    max_objects: int,
    overwrite: bool,
    detectors: Dict[str, object],
) -> dict:
    rows = _load_jsonl(manifest_path)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    enriched_rows: List[Dict[str, str]] = []
    depth_stats: List[float] = []
    flow_stats: List[float] = []
    mask_coverage: List[float] = []

    for row in rows:
        video_path = Path(row['video_path'])
        if not video_path.exists():
            raise FileNotFoundError(f'missing video: {video_path}')
        sidecar_path = sidecar_dir / f'{video_path.stem}.npz'
        if overwrite or not sidecar_path.exists():
            sidecars = _build_sidecars(
                video_path,
                num_frames=num_frames,
                image_size=image_size,
                max_objects=max_objects,
                detectors=detectors,
            )
            np.savez_compressed(sidecar_path, **sidecars)
        else:
            loaded = np.load(sidecar_path)
            sidecars = {key: loaded[key] for key in loaded.files}
        enriched = dict(row)
        enriched['tracks_path'] = str(sidecar_path)
        enriched['masks_path'] = str(sidecar_path)
        enriched['depth_path'] = str(sidecar_path)
        enriched['visibility_path'] = str(sidecar_path)
        enriched['flow_path'] = str(sidecar_path)
        enriched['occlusion_path'] = str(sidecar_path)
        enriched_rows.append(enriched)
        depth_stats.append(float(np.asarray(sidecars['depth']).mean()))
        flow_stats.append(float(np.linalg.norm(np.asarray(sidecars['flow']), axis=1).mean()))
        mask_coverage.append(float(np.asarray(sidecars['masks']).mean()))

    _write_jsonl(enriched_rows, output_manifest_path)
    meta = {
        'input_manifest': str(manifest_path),
        'output_manifest': str(output_manifest_path),
        'sidecar_dir': str(sidecar_dir),
        'num_rows': len(enriched_rows),
        'num_frames': num_frames,
        'image_size': image_size,
        'max_objects': max_objects,
        'mean_depth': float(np.mean(depth_stats)) if depth_stats else None,
        'mean_flow_magnitude': float(np.mean(flow_stats)) if flow_stats else None,
        'mean_mask_coverage': float(np.mean(mask_coverage)) if mask_coverage else None,
    }
    (output_manifest_path.parent / 'meta_sidecar.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare VidData semantic-plus sidecars from real videos.')
    parser.add_argument('--input-dir', type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--train-manifest', type=str, default='train.jsonl')
    parser.add_argument('--val-manifest', type=str, default='val.jsonl')
    parser.add_argument('--num-frames', type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument('--max-objects', type=int, default=DEFAULT_MAX_OBJECTS)
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE)
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

    if args.device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    detectors = _load_detectors(device)
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
        detectors=detectors,
    )
    val_meta = _prepare_manifest(
        val_manifest,
        val_output,
        val_sidecar_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        max_objects=args.max_objects,
        overwrite=args.overwrite,
        detectors=detectors,
    )

    combined_meta = {
        'train': train_meta,
        'val': val_meta,
    }
    (output_dir / 'meta.json').write_text(json.dumps(combined_meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(combined_meta, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
