from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from anti_chimera.text import PromptParser


@dataclass
class PlannedScene:
    tracks: torch.Tensor
    depth: torch.Tensor
    visibility: torch.Tensor
    masks: torch.Tensor
    flow: torch.Tensor
    occlusion: torch.Tensor


class PromptScenePlanner:
    def __init__(self, max_objects: int, num_frames: int, image_size: int) -> None:
        self.max_objects = max_objects
        self.num_frames = num_frames
        self.image_size = image_size
        self.parser = PromptParser()

    def _infer_action(self, prompt: str) -> str:
        p = prompt.lower()
        if any(w in p for w in ['fight', 'collid', 'crash', 'bump']):
            return 'colliding'
        if any(w in p for w in ['cross', 'pass']):
            return 'crossing'
        if any(w in p for w in ['chase', 'follow', 'pursu']):
            return 'chasing'
        if any(w in p for w in ['circle', 'orbit']):
            return 'circling'
        return 'colliding'

    def _centers(self, n: int, action: str, slot: int) -> torch.Tensor:
        T = self.num_frames
        ts = torch.linspace(0.0, 1.0, T)
        if n <= 1:
            sx, sy, tx, ty = 0.3, 0.5, 0.7, 0.5
        elif n == 2:
            starts = [(0.22, 0.5), (0.78, 0.5)]
            if action == 'crossing':
                targets = [(0.78, 0.5), (0.22, 0.5)]
            elif action == 'chasing':
                targets = [(0.72, 0.5), (0.58, 0.5)]
            else:
                targets = [(0.5, 0.5), (0.5, 0.5)]
            (sx, sy), (tx, ty) = starts[slot], targets[slot]
        else:
            starts = [(0.18, 0.5), (0.82, 0.5), (0.5, 0.2)]
            targets = [(0.5, 0.48), (0.52, 0.52), (0.5, 0.78)]
            (sx, sy), (tx, ty) = starts[slot], targets[slot]
        x = sx * (1 - ts) + tx * ts
        y = sy * (1 - ts) + ty * ts
        if action in {'colliding', 'crossing'}:
            y = y + (0.03 + 0.01 * slot) * torch.sin(torch.linspace(slot, slot + math.pi, T))
        elif action == 'circling':
            angle = torch.linspace(0.0, 2 * math.pi, T)
            radius = 0.16 + 0.03 * slot
            x = 0.5 + radius * torch.cos(angle + slot)
            y = 0.5 + radius * torch.sin(angle + slot)
        return torch.stack([x.clamp(0.08, 0.92), y.clamp(0.08, 0.92)], dim=1)

    def _tracks_from_centers(self, centers: torch.Tensor, radius: float) -> torch.Tensor:
        x = centers[:, 0]
        y = centers[:, 1]
        return torch.stack([(x - radius).clamp(0.0, 1.0), (y - radius).clamp(0.0, 1.0), (x + radius).clamp(0.0, 1.0), (y + radius).clamp(0.0, 1.0)], dim=1)

    def plan(self, prompt: str, device: torch.device) -> PlannedScene:
        n = min(max(1, len(self.parser.parse(prompt).entities)), self.max_objects)
        action = self._infer_action(prompt)
        tracks = torch.zeros(self.num_frames, self.max_objects, 4, device=device)
        radius = 0.12 if n <= 2 else 0.10
        for slot in range(n):
            tracks[:, slot] = self._tracks_from_centers(self._centers(n, action, slot).to(device), radius)
        H = self.image_size
        W = self.image_size
        masks = torch.zeros(self.num_frames, self.max_objects, H, W, device=device)
        depth = torch.ones(self.num_frames, H, W, device=device) * 0.5
        visibility = torch.zeros(self.num_frames, self.max_objects, device=device)
        flow = torch.zeros(self.num_frames, 2, H, W, device=device)
        occlusion = torch.zeros(self.num_frames, H, W, device=device)
        depth_values = torch.linspace(0.25, 0.75, steps=max(n, 2), device=device)[:n]
        for t in range(self.num_frames):
            occ = torch.zeros(H, W, device=device)
            for slot in range(n):
                x1, y1, x2, y2 = tracks[t, slot]
                ix1 = max(0, min(W - 1, int(math.floor(float(x1) * W))))
                iy1 = max(0, min(H - 1, int(math.floor(float(y1) * H))))
                ix2 = max(ix1 + 1, min(W, int(math.ceil(float(x2) * W))))
                iy2 = max(iy1 + 1, min(H, int(math.ceil(float(y2) * H))))
                masks[t, slot, iy1:iy2, ix1:ix2] = 1.0
                visibility[t, slot] = 1.0
                depth[t, iy1:iy2, ix1:ix2] = torch.minimum(depth[t, iy1:iy2, ix1:ix2], torch.full((iy2 - iy1, ix2 - ix1), float(depth_values[slot]), device=device))
                occ[iy1:iy2, ix1:ix2] += 1.0
                if t + 1 < self.num_frames:
                    dx = ((tracks[t + 1, slot, 0] + tracks[t + 1, slot, 2]) - (tracks[t, slot, 0] + tracks[t, slot, 2])) * 0.5 * W
                    dy = ((tracks[t + 1, slot, 1] + tracks[t + 1, slot, 3]) - (tracks[t, slot, 1] + tracks[t, slot, 3])) * 0.5 * H
                    mask = masks[t, slot] > 0
                    flow[t, 0][mask] = dx / max(H, W)
                    flow[t, 1][mask] = dy / max(H, W)
            occlusion[t] = (occ > 1.0).float()
        if self.num_frames > 1:
            flow[-1] = flow[-2]
        return PlannedScene(tracks=tracks, depth=depth, visibility=visibility, masks=masks, flow=flow, occlusion=occlusion)
