from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch


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
    if device_name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def normalize_video(video: torch.Tensor) -> torch.Tensor:
    return video * 2.0 - 1.0


def denormalize_video(video: torch.Tensor) -> torch.Tensor:
    return ((video + 1.0) * 0.5).clamp(0.0, 1.0)


def save_gif(video: torch.Tensor, path: str | Path, fps: int = 6) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = denormalize_video(video.detach().cpu())
    frames = []
    for t in range(v.shape[1]):
        frame = (v[:, t].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        frames.append(frame)
    imageio.mimsave(path, frames, fps=fps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999).float()
