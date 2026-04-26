from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from anti_chimera.text import PromptParser


class SceneHintBuilder:
    def __init__(self, max_objects: int, depth_bins: int, image_size: int) -> None:
        self.max_objects = max_objects
        self.depth_bins = depth_bins
        self.image_size = image_size
        self.parser = PromptParser()

    def _build_entity_grounding(self, tracks: torch.Tensor, out_size: int) -> torch.Tensor:
        T, K, _ = tracks.shape
        grid = torch.zeros(T, K, out_size, out_size, dtype=torch.float32, device=tracks.device)
        for t in range(T):
            for k in range(K):
                x1, y1, x2, y2 = tracks[t, k]
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue
                ix1 = int(torch.floor(x1 * out_size).item())
                iy1 = int(torch.floor(y1 * out_size).item())
                ix2 = int(torch.ceil(x2 * out_size).item())
                iy2 = int(torch.ceil(y2 * out_size).item())
                ix1, iy1 = max(0, ix1), max(0, iy1)
                ix2, iy2 = min(out_size, ix2), min(out_size, iy2)
                grid[t, k, iy1:iy2, ix1:ix2] = 1.0
        return grid

    def _build_mask_grounding(self, masks: torch.Tensor, out_size: int) -> torch.Tensor:
        if masks.ndim != 4:
            raise ValueError(f'expected masks with 4 dims, got {tuple(masks.shape)}')
        return F.interpolate(masks.float(), size=(out_size, out_size), mode='bilinear', align_corners=False)

    def _build_depth_bins(self, depth: torch.Tensor, out_size: int) -> torch.Tensor:
        depth_ds = F.interpolate(depth[:, None], size=(out_size, out_size), mode='bilinear', align_corners=False)[:, 0]
        bins = torch.linspace(0.0, 1.0, self.depth_bins + 1, device=depth.device)
        out = torch.zeros(depth_ds.shape[0], self.depth_bins, out_size, out_size, device=depth.device)
        for b in range(self.depth_bins):
            left, right = bins[b], bins[b + 1]
            mask = (depth_ds >= left) & (depth_ds < right)
            out[:, b][mask] = 1.0
        out[:, -1] = torch.where(out.sum(dim=1) == 0, torch.ones_like(out[:, -1]), out[:, -1])
        return out

    def _build_visibility_map(self, visibility: torch.Tensor, out_size: int) -> torch.Tensor:
        T, K = visibility.shape
        vis = visibility[:, :, None, None].expand(T, K, out_size, out_size)
        return vis.float()

    def _build_flow_map(self, flow: torch.Tensor, out_size: int) -> torch.Tensor:
        if flow.ndim != 4:
            raise ValueError(f'expected flow with 4 dims, got {tuple(flow.shape)}')
        return F.interpolate(flow.float(), size=(out_size, out_size), mode='bilinear', align_corners=False)

    def _build_occlusion_map(self, occlusion: torch.Tensor, out_size: int) -> torch.Tensor:
        if occlusion.ndim != 3:
            raise ValueError(f'expected occlusion with 3 dims, got {tuple(occlusion.shape)}')
        return F.interpolate(occlusion[:, None].float(), size=(out_size, out_size), mode='bilinear', align_corners=False)

    def build(self, sample: Dict[str, torch.Tensor | str | List[str]]) -> torch.Tensor:
        video = sample['video']
        T, _, H, W = video.permute(1, 0, 2, 3).shape
        out_size = H
        device = video.device
        tracks = sample.get('tracks')
        depth = sample.get('depth')
        visibility = sample.get('visibility')
        masks = sample.get('masks')
        flow = sample.get('flow')
        occlusion = sample.get('occlusion')
        if tracks is None:
            tracks = torch.zeros(T, self.max_objects, 4, device=device)
        if depth is None:
            depth = torch.full((T, self.image_size, self.image_size), 0.5, device=device)
        if visibility is None:
            visibility = torch.zeros(T, self.max_objects, device=device)
        if masks is None:
            masks = torch.zeros(T, self.max_objects, self.image_size, self.image_size, device=device)
        if flow is None:
            flow = torch.zeros(T, 2, self.image_size, self.image_size, device=device)
        if occlusion is None:
            occlusion = torch.zeros(T, self.image_size, self.image_size, device=device)
        grounding = self._build_entity_grounding(tracks.float(), out_size)
        mask_grounding = self._build_mask_grounding(masks.float(), out_size)
        depth_bins = self._build_depth_bins(depth.float(), out_size)
        vis_map = self._build_visibility_map(visibility.float(), out_size)
        flow_map = self._build_flow_map(flow.float(), out_size)
        occlusion_map = self._build_occlusion_map(occlusion.float(), out_size)
        cond = torch.cat([
            grounding.reshape(T, -1, out_size, out_size),
            mask_grounding.reshape(T, -1, out_size, out_size),
            depth_bins.reshape(T, -1, out_size, out_size),
            vis_map.reshape(T, -1, out_size, out_size),
            flow_map.reshape(T, -1, out_size, out_size),
            occlusion_map.reshape(T, -1, out_size, out_size),
        ], dim=1)
        return cond.permute(1, 0, 2, 3).contiguous()
