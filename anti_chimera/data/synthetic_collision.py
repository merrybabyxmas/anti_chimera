from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


COLORS = {
    'white': np.array([1.0, 1.0, 1.0], dtype=np.float32),
    'black': np.array([0.05, 0.05, 0.05], dtype=np.float32),
    'red': np.array([0.95, 0.15, 0.15], dtype=np.float32),
    'blue': np.array([0.15, 0.35, 0.95], dtype=np.float32),
    'green': np.array([0.15, 0.85, 0.25], dtype=np.float32),
    'yellow': np.array([0.95, 0.85, 0.15], dtype=np.float32),
}
SHAPES = ['circle', 'square', 'triangle']
ACTIONS = ['colliding', 'crossing', 'chasing']


@dataclass
class ObjectSpec:
    color_name: str
    shape_name: str
    radius: float
    z_depth: float
    trajectory: np.ndarray


class SyntheticCollisionDataset(Dataset):
    def __init__(
        self,
        size: int,
        num_frames: int,
        image_size: int,
        max_objects: int = 3,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.size = size
        self.num_frames = num_frames
        self.image_size = image_size
        self.max_objects = max_objects
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def _rng(self, idx: int) -> random.Random:
        return random.Random(self.seed * 100003 + idx)

    def _sample_trajectory(self, rng: random.Random, T: int) -> np.ndarray:
        start = np.array([rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)], dtype=np.float32)
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(0.20, 0.45)
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        end = np.clip(start + direction * speed, 0.1, 0.9)
        ts = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
        path = start[None, :] * (1 - ts) + end[None, :] * ts
        wiggle = rng.uniform(0.0, 0.03)
        if wiggle > 0:
            perp = np.array([-direction[1], direction[0]], dtype=np.float32)
            phase = rng.uniform(0, 2 * math.pi)
            sinus = np.sin(np.linspace(phase, phase + 2 * math.pi, T, dtype=np.float32))[:, None]
            path += wiggle * sinus * perp[None, :]
        return np.clip(path, 0.05, 0.95)

    def _make_collision_paths(self, rng: random.Random, T: int) -> Tuple[np.ndarray, np.ndarray]:
        center = np.array([rng.uniform(0.4, 0.6), rng.uniform(0.4, 0.6)], dtype=np.float32)
        direction = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)], dtype=np.float32)
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        offset = rng.uniform(0.22, 0.35)
        a0 = np.clip(center - direction * offset, 0.1, 0.9)
        b0 = np.clip(center + direction * offset, 0.1, 0.9)
        ts = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
        curve = np.sin(np.linspace(0, math.pi, T, dtype=np.float32))[:, None] * rng.uniform(0.0, 0.06)
        perp = np.array([-direction[1], direction[0]], dtype=np.float32)
        a = a0[None, :] * (1 - ts) + center[None, :] * ts + curve * perp[None, :]
        b = b0[None, :] * (1 - ts) + center[None, :] * ts - curve * perp[None, :]
        return np.clip(a, 0.05, 0.95), np.clip(b, 0.05, 0.95)

    def _render_object(self, frame: np.ndarray, inst: np.ndarray, depth: np.ndarray, obj_id: int, spec: ObjectSpec, xy: np.ndarray):
        H = frame.shape[0]
        W = frame.shape[1]
        cy = xy[1] * (H - 1)
        cx = xy[0] * (W - 1)
        r = spec.radius * H
        yy, xx = np.mgrid[0:H, 0:W]
        dx = xx - cx
        dy = yy - cy
        if spec.shape_name == 'circle':
            mask = dx * dx + dy * dy <= r * r
        elif spec.shape_name == 'square':
            mask = (np.abs(dx) <= r) & (np.abs(dy) <= r)
        else:
            mask = (np.abs(dx) / (r + 1e-6) + np.maximum(dy, -r) / (r + 1e-6) <= 1.0) & (dy >= -r)
        overwrite = mask & (spec.z_depth <= depth)
        frame[overwrite] = COLORS[spec.color_name]
        inst[overwrite] = obj_id
        depth[overwrite] = spec.z_depth

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | List[str]]:
        rng = self._rng(idx)
        T = self.num_frames
        H = self.image_size
        W = self.image_size
        n_obj = rng.choice([2, 2, 2, 3]) if self.max_objects >= 3 else 2
        colors = rng.sample(list(COLORS.keys()), n_obj)
        shapes = [rng.choice(SHAPES) for _ in range(n_obj)]
        action = rng.choice(ACTIONS)
        specs: List[ObjectSpec] = []
        if action == 'colliding':
            a, b = self._make_collision_paths(rng, T)
            trajs = [a, b]
            if n_obj == 3:
                trajs.append(self._sample_trajectory(rng, T))
        else:
            trajs = [self._sample_trajectory(rng, T) for _ in range(n_obj)]
        z_depths = sorted([rng.uniform(0.2, 0.8) for _ in range(n_obj)])
        rng.shuffle(z_depths)
        for i in range(n_obj):
            specs.append(ObjectSpec(
                color_name=colors[i],
                shape_name=shapes[i],
                radius=rng.uniform(0.08, 0.13),
                z_depth=float(z_depths[i]),
                trajectory=trajs[i],
            ))

        video = np.zeros((T, H, W, 3), dtype=np.float32)
        instance_map = np.full((T, H, W), -1, dtype=np.int64)
        depth_map = np.ones((T, H, W), dtype=np.float32)
        tracks = np.zeros((T, self.max_objects, 4), dtype=np.float32)
        visibility = np.zeros((T, self.max_objects), dtype=np.float32)

        for t in range(T):
            frame = np.full((H, W, 3), 0.55, dtype=np.float32)
            inst = np.full((H, W), -1, dtype=np.int64)
            depth = np.ones((H, W), dtype=np.float32)
            draw_order = sorted(list(enumerate(specs)), key=lambda x: x[1].z_depth, reverse=True)
            for obj_id, spec in draw_order:
                xy = spec.trajectory[t]
                self._render_object(frame, inst, depth, obj_id, spec, xy)
            video[t] = frame
            instance_map[t] = inst
            depth_map[t] = depth
            for obj_id, spec in enumerate(specs):
                xy = spec.trajectory[t]
                r = spec.radius
                tracks[t, obj_id] = np.array([
                    max(0.0, xy[0] - r), max(0.0, xy[1] - r),
                    min(1.0, xy[0] + r), min(1.0, xy[1] + r),
                ], dtype=np.float32)
                visibility[t, obj_id] = float((inst == obj_id).sum() > 0)

        entities = [f"{spec.color_name} {spec.shape_name}" for spec in specs]
        if len(entities) == 2:
            caption = f"a {entities[0]} and a {entities[1]} {action}"
        else:
            caption = f"a {entities[0]}, a {entities[1]}, and a {entities[2]} {action}"

        return {
            'video': torch.from_numpy(video).permute(3, 0, 1, 2).contiguous(),
            'caption': caption,
            'entities': entities,
            'tracks': torch.from_numpy(tracks),
            'depth': torch.from_numpy(depth_map),
            'instance_map': torch.from_numpy(instance_map),
            'visibility': torch.from_numpy(visibility),
        }
