from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cycle(dl: Iterable):
    while True:
        for batch in dl:
            yield batch


def default_device(device_name: str) -> torch.device:
    name = str(device_name).lower()
    if name == 'cpu':
        return torch.device('cpu')
    if name == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if name in {'cuda', 'auto'}:
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device('cpu')


def normalize_video(video: torch.Tensor) -> torch.Tensor:
    return video * 2.0 - 1.0


def denormalize_video(video: torch.Tensor) -> torch.Tensor:
    return ((video + 1.0) * 0.5).clamp(0.0, 1.0)


def _video_frames(video: torch.Tensor) -> list[np.ndarray]:
    v = denormalize_video(video.detach().cpu())
    frames = []
    for t in range(v.shape[1]):
        frame = (v[:, t].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        frames.append(frame)
    return frames


def save_gif(video: torch.Tensor, path: str | Path, fps: int = 6) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = _video_frames(video)
    imageio.mimsave(path, frames, fps=fps)


def save_video_png(video: torch.Tensor, path: str | Path, max_frames: int = 8) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = _video_frames(video)
    if not frames:
        raise ValueError('video has no frames to render')
    count = min(max_frames, len(frames))
    idxs = np.linspace(0, len(frames) - 1, count).round().astype(int)
    selected = [frames[i] for i in idxs]
    cols = min(4, len(selected))
    rows = math.ceil(len(selected) / cols)
    tile_h, tile_w, _ = selected[0].shape
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, frame in enumerate(selected):
        r, c = divmod(i, cols)
        canvas[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = frame
    Image.fromarray(canvas).save(path)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999).float()
