from __future__ import annotations

import torch
import torch.nn.functional as F


class MinimalSceneHintBuilder:
    def __init__(self, max_objects: int, depth_bins: int, image_size: int) -> None:
        self.max_objects = int(max_objects)
        self.depth_bins = int(depth_bins)
        self.image_size = int(image_size)

    def num_channels(self) -> int:
        return self.max_objects + self.depth_bins + self.max_objects + 1

    def _boxes(self, tracks: torch.Tensor, out_size: int) -> torch.Tensor:
        T, K, _ = tracks.shape
        grid = torch.zeros(T, K, out_size, out_size, dtype=torch.float32, device=tracks.device)
        for t in range(T):
            for k in range(K):
                x1, y1, x2, y2 = tracks[t, k]
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue
                ix1 = max(0, int(torch.floor(x1 * out_size).item()))
                iy1 = max(0, int(torch.floor(y1 * out_size).item()))
                ix2 = min(out_size, int(torch.ceil(x2 * out_size).item()))
                iy2 = min(out_size, int(torch.ceil(y2 * out_size).item()))
                grid[t, k, iy1:iy2, ix1:ix2] = 1.0
        return grid

    def _depth_bins(self, depth: torch.Tensor, out_size: int) -> torch.Tensor:
        depth_ds = F.interpolate(depth[:, None], size=(out_size, out_size), mode='bilinear', align_corners=False)[:, 0]
        bins = torch.linspace(0.0, 1.0, self.depth_bins + 1, device=depth.device)
        out = torch.zeros(depth_ds.shape[0], self.depth_bins, out_size, out_size, device=depth.device)
        for b in range(self.depth_bins):
            out[:, b] = ((depth_ds >= bins[b]) & (depth_ds < bins[b + 1])).float()
        out[:, -1] = torch.where(out.sum(dim=1) == 0, torch.ones_like(out[:, -1]), out[:, -1])
        return out

    def _visibility(self, visibility: torch.Tensor, out_size: int) -> torch.Tensor:
        return visibility[:, :, None, None].expand(-1, -1, out_size, out_size).float()

    def _overlap(self, grounding: torch.Tensor) -> torch.Tensor:
        return ((grounding > 0.15).float().sum(dim=1, keepdim=True) > 1.0).float()

    def build(self, sample):
        video = sample['video']
        T, _, H, _ = video.permute(1, 0, 2, 3).shape
        device = video.device
        tracks = sample.get('tracks')
        depth = sample.get('depth')
        visibility = sample.get('visibility')
        if tracks is None:
            tracks = torch.zeros(T, self.max_objects, 4, device=device)
        if depth is None:
            depth = torch.full((T, self.image_size, self.image_size), 0.5, device=device)
        if visibility is None:
            visibility = torch.zeros(T, self.max_objects, device=device)
        grounding = self._boxes(tracks.float().to(device), H)
        depth_bins = self._depth_bins(depth.float().to(device), H)
        vis_map = self._visibility(visibility.float().to(device), H)
        overlap = self._overlap(grounding)
        cond = torch.cat([
            grounding.reshape(T, -1, H, H),
            depth_bins.reshape(T, -1, H, H),
            vis_map.reshape(T, -1, H, H),
            overlap.reshape(T, -1, H, H),
        ], dim=1)
        return cond.permute(1, 0, 2, 3).contiguous()
