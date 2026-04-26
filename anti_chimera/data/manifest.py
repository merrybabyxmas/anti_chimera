from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def _load_jsonl(path: str | Path) -> List[Dict]:
    items: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _resize_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    img = Image.fromarray(frame.astype(np.uint8))
    img = img.resize((image_size, image_size), resample=Image.BICUBIC)
    return np.asarray(img)


class ManifestVideoDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path,
        num_frames: int,
        image_size: int,
        max_objects: int,
    ) -> None:
        self.items = _load_jsonl(manifest_path)
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.max_objects = max_objects

    def __len__(self) -> int:
        return len(self.items)

    def _resolve(self, rel: str | None) -> Path | None:
        if rel is None:
            return None
        path = Path(rel)
        if path.is_absolute():
            return path
        return self.root_dir / path

    def _load_video(self, path: Path) -> torch.Tensor:
        reader = imageio.get_reader(path)
        frames = []
        try:
            while True:
                try:
                    frame = reader.get_next_data()
                except (IndexError, StopIteration):
                    break
                if frame.ndim == 2:
                    frame = np.repeat(frame[..., None], 3, axis=2)
                frames.append(_resize_frame(frame[..., :3], self.image_size))
        finally:
            reader.close()
        if not frames:
            raise RuntimeError(f'No frames found in {path}')
        idxs = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
        sampled = np.stack([frames[i] for i in idxs], axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(sampled).permute(3, 0, 1, 2).contiguous()

    def _load_optional_array(self, path: Path | None, shape: tuple, dtype=np.float32, key: str | None = None) -> torch.Tensor:
        if path is None or not path.exists():
            return torch.zeros(shape, dtype=torch.float32)
        if path.suffix == '.npy':
            arr = np.load(path)
        elif path.suffix == '.npz':
            loaded = np.load(path)
            if key is not None and key in loaded.files:
                arr = loaded[key]
            else:
                arr = loaded[loaded.files[0]]
        else:
            raise ValueError(f'Unsupported sidecar format: {path}')
        arr = arr.astype(dtype)
        return torch.from_numpy(arr)

    def _resample_time(self, tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
        if tensor.shape[0] == target_frames:
            return tensor
        idxs = np.linspace(0, tensor.shape[0] - 1, target_frames).round().astype(int)
        return tensor[torch.from_numpy(idxs).long()]

    def _resize_spatial(self, tensor: torch.Tensor, image_size: int) -> torch.Tensor:
        if tensor.ndim == 3:
            if tensor.shape[-2:] == (image_size, image_size):
                return tensor
            resized = F.interpolate(tensor[:, None].float(), size=(image_size, image_size), mode='bilinear', align_corners=False)
            return resized[:, 0]
        if tensor.ndim == 4:
            if tensor.shape[-2:] == (image_size, image_size):
                return tensor
            leading = tensor.shape[:-2]
            flat = tensor.reshape(-1, 1, tensor.shape[-2], tensor.shape[-1]).float()
            resized = F.interpolate(flat, size=(image_size, image_size), mode='bilinear', align_corners=False)
            return resized.reshape(*leading, image_size, image_size)
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | List[str]]:
        item = self.items[idx]
        video = self._load_video(self._resolve(item['video_path']))
        T = video.shape[1]
        H = video.shape[2]
        W = video.shape[3]
        tracks = self._load_optional_array(self._resolve(item.get('tracks_path')), (T, self.max_objects, 4), key='tracks')
        depth = self._load_optional_array(self._resolve(item.get('depth_path')), (T, H, W), key='depth')
        visibility = self._load_optional_array(self._resolve(item.get('visibility_path')), (T, self.max_objects), key='visibility')
        masks = self._load_optional_array(self._resolve(item.get('masks_path')), (T, self.max_objects, H, W), key='masks')
        flow = self._load_optional_array(self._resolve(item.get('flow_path')), (T, 2, H, W), key='flow')
        occlusion = self._load_optional_array(self._resolve(item.get('occlusion_path')), (T, H, W), key='occlusion')
        tracks = self._resample_time(tracks, T)
        depth = self._resize_spatial(self._resample_time(depth, T), H)
        visibility = self._resample_time(visibility, T)
        masks = self._resize_spatial(self._resample_time(masks, T), H)
        flow = self._resize_spatial(self._resample_time(flow, T), H)
        occlusion = self._resize_spatial(self._resample_time(occlusion, T), H)
        return {
            'video': video,
            'caption': item['caption'],
            'entities': item.get('entities', []),
            'tracks': tracks,
            'depth': depth,
            'visibility': visibility,
            'masks': masks,
            'flow': flow,
            'occlusion': occlusion,
        }
