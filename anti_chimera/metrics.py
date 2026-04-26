from __future__ import annotations

from typing import Dict

import torch


def _track_to_box(track: torch.Tensor, height: int, width: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = track.tolist()
    ix1 = max(0, min(width - 1, int(round(x1 * (width - 1)))))
    iy1 = max(0, min(height - 1, int(round(y1 * (height - 1)))))
    ix2 = max(ix1 + 1, min(width, int(round(x2 * (width - 1))) + 1))
    iy2 = max(iy1 + 1, min(height, int(round(y2 * (height - 1))) + 1))
    return ix1, iy1, ix2, iy2


def _crop_mean(video: torch.Tensor, track: torch.Tensor, frame_idx: int) -> torch.Tensor:
    _, _, height, width = video.shape
    ix1, iy1, ix2, iy2 = _track_to_box(track, height, width)
    crop = video[:, frame_idx, iy1:iy2, ix1:ix2]
    if crop.numel() == 0:
        return torch.zeros(3, dtype=video.dtype, device=video.device)
    return crop.reshape(3, -1).mean(dim=1)


def _pairwise_iou(track_a: torch.Tensor, track_b: torch.Tensor) -> float:
    ax1, ay1, ax2, ay2 = track_a.tolist()
    bx1, by1, bx2, by2 = track_b.tolist()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 1e-8:
        return 0.0
    return float(inter / denom)


def compute_chimera_metrics(
    target_video: torch.Tensor,
    generated_video: torch.Tensor,
    tracks: torch.Tensor,
    visibility: torch.Tensor,
) -> Dict[str, float]:
    """Compute lightweight chimera-oriented metrics.

    Parameters
    ----------
    target_video : [C, T, H, W] in [0, 1]
    generated_video : [C, T, H, W] in [0, 1]
    tracks : [T, K, 4] normalized xyxy
    visibility : [T, K]
    """
    target_video = target_video.detach().float()
    generated_video = generated_video.detach().float()
    tracks = tracks.detach().float()
    visibility = visibility.detach().float()

    _, T, _, _ = target_video.shape
    K = tracks.shape[1]
    ref_colors = []
    for k in range(K):
        first_visible = None
        for t in range(T):
            if visibility[t, k] > 0.5:
                first_visible = t
                break
        if first_visible is None:
            ref_colors.append(torch.zeros(3, device=target_video.device))
        else:
            ref_colors.append(_crop_mean(target_video, tracks[first_visible, k], first_visible))
    ref_colors = torch.stack(ref_colors, dim=0)

    identity_errors = []
    overlap_errors = []
    overlap_ious = []
    separation_scores = []
    visible_count = 0
    overlap_count = 0

    for t in range(T):
        frame_colors = []
        for k in range(K):
            if visibility[t, k] > 0.5:
                visible_count += 1
                frame_color = _crop_mean(generated_video, tracks[t, k], t)
                frame_colors.append(frame_color)
                identity_errors.append(torch.mean(torch.abs(frame_color - ref_colors[k])).item())
            else:
                frame_colors.append(torch.zeros(3, device=generated_video.device))
        frame_colors = torch.stack(frame_colors, dim=0)

        frame_has_overlap = False
        for i in range(K):
            for j in range(i + 1, K):
                iou = _pairwise_iou(tracks[t, i], tracks[t, j])
                if iou > 0.05:
                    frame_has_overlap = True
                    overlap_ious.append(iou)
                    sep = torch.mean(torch.abs(frame_colors[i] - frame_colors[j])).item()
                    separation_scores.append(sep)
        if frame_has_overlap:
            overlap_count += 1
            for k in range(K):
                if visibility[t, k] > 0.5:
                    overlap_errors.append(torch.mean(torch.abs(frame_colors[k] - ref_colors[k])).item())

    return {
        'identity_l1': float(sum(identity_errors) / max(len(identity_errors), 1)),
        'overlap_identity_l1': float(sum(overlap_errors) / max(len(overlap_errors), 1)),
        'mean_overlap_iou': float(sum(overlap_ious) / max(len(overlap_ious), 1)),
        'color_separation_l1': float(sum(separation_scores) / max(len(separation_scores), 1)),
        'visible_instances': float(visible_count),
        'overlap_frames': float(overlap_count),
    }
