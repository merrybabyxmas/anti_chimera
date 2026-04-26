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
ACTIONS = ['colliding', 'crossing', 'chasing', 'circling']


@dataclass
class ObjectSpec:
    color_name: str
    shape_name: str
    radius: float
    z_depth: float
    trajectory: np.ndarray


class SyntheticCollisionDataset(Dataset):
    """A structured synthetic dataset for anti-chimera research.

    Compared with the earlier version, this dataset now returns additional dense
    cues used by the scene-hint builder:

    - visible masks
    - amodal masks
    - dense optical flow
    - occlusion map
    - overlap rate / difficulty metadata

    The dataset also supports a lightweight curriculum via `difficulty`.
    """

    def __init__(
        self,
        size: int,
        num_frames: int,
        image_size: int,
        max_objects: int = 3,
        seed: int = 42,
        difficulty: str = 'mixed',
    ) -> None:
        super().__init__()
        self.size = size
        self.num_frames = num_frames
        self.image_size = image_size
        self.max_objects = max_objects
        self.seed = seed
        self.difficulty = difficulty

    def __len__(self) -> int:
        return self.size

    def _rng(self, idx: int) -> random.Random:
        return random.Random(self.seed * 100003 + idx)

    def _difficulty_profile(self, rng: random.Random) -> dict:
        difficulty = self.difficulty
        if difficulty == 'mixed':
            difficulty = rng.choice(['easy', 'medium', 'hard'])
        if difficulty == 'easy':
            return {'objects': [2], 'overlap_scale': 0.10, 'wiggle': (0.0, 0.02), 'radius': (0.08, 0.11)}
        if difficulty == 'medium':
            return {'objects': [2, 3], 'overlap_scale': 0.18, 'wiggle': (0.0, 0.035), 'radius': (0.09, 0.12)}
        return {'objects': [3], 'overlap_scale': 0.28, 'wiggle': (0.01, 0.05), 'radius': (0.10, 0.14)}

    def _sample_trajectory(self, rng: random.Random, T: int, wiggle_range: Tuple[float, float]) -> np.ndarray:
        start = np.array([rng.uniform(0.18, 0.82), rng.uniform(0.18, 0.82)], dtype=np.float32)
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(0.18, 0.45)
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        end = np.clip(start + direction * speed, 0.08, 0.92)
        ts = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
        path = start[None, :] * (1 - ts) + end[None, :] * ts
        wiggle = rng.uniform(*wiggle_range)
        if wiggle > 0:
            perp = np.array([-direction[1], direction[0]], dtype=np.float32)
            phase = rng.uniform(0, 2 * math.pi)
            sinus = np.sin(np.linspace(phase, phase + 2 * math.pi, T, dtype=np.float32))[:, None]
            path += wiggle * sinus * perp[None, :]
        return np.clip(path, 0.05, 0.95)

    def _make_collision_paths(self, rng: random.Random, T: int, overlap_scale: float) -> Tuple[np.ndarray, np.ndarray]:
        center = np.array([rng.uniform(0.4, 0.6), rng.uniform(0.4, 0.6)], dtype=np.float32)
        direction = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)], dtype=np.float32)
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        offset = rng.uniform(0.16, 0.32)
        a0 = np.clip(center - direction * offset, 0.1, 0.9)
        b0 = np.clip(center + direction * offset, 0.1, 0.9)
        ts = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
        curve = np.sin(np.linspace(0, math.pi, T, dtype=np.float32))[:, None] * rng.uniform(0.0, overlap_scale)
        perp = np.array([-direction[1], direction[0]], dtype=np.float32)
        a = a0[None, :] * (1 - ts) + center[None, :] * ts + curve * perp[None, :]
        b = b0[None, :] * (1 - ts) + center[None, :] * ts - curve * perp[None, :]
        return np.clip(a, 0.05, 0.95), np.clip(b, 0.05, 0.95)

    def _shape_mask(self, spec: ObjectSpec, xy: np.ndarray, H: int, W: int) -> np.ndarray:
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
        return mask

    def _render_frame(
        self,
        specs: List[ObjectSpec],
        positions: List[np.ndarray],
        H: int,
        W: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        background = np.linspace(0.48, 0.62, H, dtype=np.float32)[:, None]
        frame = np.stack([background.repeat(W, axis=1) for _ in range(3)], axis=-1)
        inst = np.full((H, W), -1, dtype=np.int64)
        depth = np.ones((H, W), dtype=np.float32)
        visible_masks = np.zeros((len(specs), H, W), dtype=np.float32)
        amodal_masks = np.zeros((len(specs), H, W), dtype=np.float32)

        order = sorted(list(enumerate(specs)), key=lambda x: x[1].z_depth, reverse=True)
        for obj_id, spec in order:
            amodal = self._shape_mask(spec, positions[obj_id], H, W)
            amodal_masks[obj_id] = amodal.astype(np.float32)
            overwrite = amodal & (spec.z_depth <= depth)
            frame[overwrite] = COLORS[spec.color_name]
            inst[overwrite] = obj_id
            depth[overwrite] = spec.z_depth
        for obj_id in range(len(specs)):
            visible_masks[obj_id] = (inst == obj_id).astype(np.float32)
        occlusion = ((amodal_masks.sum(axis=0) > 1.0) | ((amodal_masks.sum(axis=0) > visible_masks.sum(axis=0) + 1e-5))).astype(np.float32)
        return frame, inst, depth, visible_masks, occlusion

    def _dense_flow(self, specs: List[ObjectSpec], positions: List[np.ndarray], next_positions: List[np.ndarray], masks: np.ndarray, H: int, W: int) -> np.ndarray:
        flow = np.zeros((2, H, W), dtype=np.float32)
        for obj_id, spec in enumerate(specs):
            delta = next_positions[obj_id] - positions[obj_id]
            dx = delta[0] * W
            dy = delta[1] * H
            obj_mask = masks[obj_id] > 0.0
            flow[0, obj_mask] = dx
            flow[1, obj_mask] = dy
        return flow / max(H, W)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | List[str] | float]:
        rng = self._rng(idx)
        profile = self._difficulty_profile(rng)
        T = self.num_frames
        H = self.image_size
        W = self.image_size
        n_obj = rng.choice(profile['objects']) if self.max_objects >= 3 else 2
        colors = rng.sample(list(COLORS.keys()), n_obj)
        shapes = [rng.choice(SHAPES) for _ in range(n_obj)]
        action = rng.choice(ACTIONS)
        specs: List[ObjectSpec] = []
        if action == 'colliding':
            a, b = self._make_collision_paths(rng, T, profile['overlap_scale'])
            trajs = [a, b]
            if n_obj == 3:
                trajs.append(self._sample_trajectory(rng, T, profile['wiggle']))
        else:
            trajs = [self._sample_trajectory(rng, T, profile['wiggle']) for _ in range(n_obj)]
        z_depths = sorted([rng.uniform(0.15, 0.85) for _ in range(n_obj)])
        rng.shuffle(z_depths)
        for i in range(n_obj):
            specs.append(
                ObjectSpec(
                    color_name=colors[i],
                    shape_name=shapes[i],
                    radius=rng.uniform(*profile['radius']),
                    z_depth=float(z_depths[i]),
                    trajectory=trajs[i],
                )
            )

        video = np.zeros((T, H, W, 3), dtype=np.float32)
        instance_map = np.full((T, H, W), -1, dtype=np.int64)
        depth_map = np.ones((T, H, W), dtype=np.float32)
        tracks = np.zeros((T, self.max_objects, 4), dtype=np.float32)
        visibility = np.zeros((T, self.max_objects), dtype=np.float32)
        masks = np.zeros((T, self.max_objects, H, W), dtype=np.float32)
        amodal_masks = np.zeros((T, self.max_objects, H, W), dtype=np.float32)
        flow = np.zeros((T, 2, H, W), dtype=np.float32)
        occlusion = np.zeros((T, H, W), dtype=np.float32)
        overlap_rates: List[float] = []

        for t in range(T):
            positions = [spec.trajectory[t] for spec in specs]
            frame, inst, depth, visible_masks, occ = self._render_frame(specs, positions, H, W)
            video[t] = frame
            instance_map[t] = inst
            depth_map[t] = depth
            masks[t, :n_obj] = visible_masks
            occlusion[t] = occ
            overlap_rates.append(float(occ.mean()))

            for obj_id, spec in enumerate(specs):
                xy = positions[obj_id]
                r = spec.radius
                tracks[t, obj_id] = np.array(
                    [
                        max(0.0, xy[0] - r),
                        max(0.0, xy[1] - r),
                        min(1.0, xy[0] + r),
                        min(1.0, xy[1] + r),
                    ],
                    dtype=np.float32,
                )
                visibility[t, obj_id] = float(visible_masks[obj_id].sum() > 0)
                amodal_masks[t, obj_id] = self._shape_mask(spec, xy, H, W).astype(np.float32)

            if t + 1 < T:
                next_positions = [spec.trajectory[t + 1] for spec in specs]
                flow[t] = self._dense_flow(specs, positions, next_positions, visible_masks, H, W)
            elif T > 1:
                flow[t] = flow[t - 1]

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
            'masks': torch.from_numpy(masks),
            'amodal_masks': torch.from_numpy(amodal_masks),
            'flow': torch.from_numpy(flow),
            'occlusion': torch.from_numpy(occlusion),
            'difficulty': self.difficulty,
            'overlap_rate': float(sum(overlap_rates) / max(len(overlap_rates), 1)),
        }
