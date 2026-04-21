from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np
import torch
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
        for frame in reader:
            frames.append(_resize_frame(frame[..., :3], self.image_size))
        if not frames:
            raise RuntimeError(f'No frames found in {path}')
        idxs = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
        sampled = np.stack([frames[i] for i in idxs], axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(sampled).permute(3, 0, 1, 2).contiguous()

    def _load_optional_array(self, path: Path | None, shape: tuple, dtype=np.float32) -> torch.Tensor:
        if path is None or not path.exists():
            return torch.zeros(shape, dtype=torch.float32)
        if path.suffix == '.npy':
            arr = np.load(path)
        elif path.suffix == '.npz':
            loaded = np.load(path)
            arr = loaded[loaded.files[0]]
        else:
            raise ValueError(f'Unsupported sidecar format: {path}')
        arr = arr.astype(dtype)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | List[str]]:
        item = self.items[idx]
        video = self._load_video(self._resolve(item['video_path']))
        T = video.shape[1]
        H = video.shape[2]
        W = video.shape[3]
        tracks = self._load_optional_array(self._resolve(item.get('tracks_path')), (T, self.max_objects, 4))
        depth = self._load_optional_array(self._resolve(item.get('depth_path')), (T, H, W))
        visibility = self._load_optional_array(self._resolve(item.get('visibility_path')), (T, self.max_objects))
        return {
            'video': video,
            'caption': item['caption'],
            'entities': item.get('entities', []),
            'tracks': tracks,
            'depth': depth,
            'visibility': visibility,
        }
