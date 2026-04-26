from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from anti_chimera.text import PromptParser, SimplePromptEncoder


@dataclass
class PlannerPrediction:
    tracks: torch.Tensor          # [B, T, K, 4]
    depth_slots: torch.Tensor     # [B, T, K]
    visibility_logits: torch.Tensor  # [B, T, K]
    masks: torch.Tensor           # [B, T, K, H, W]
    depth: torch.Tensor           # [B, T, H, W]
    visibility: torch.Tensor      # [B, T, K]
    flow: torch.Tensor            # [B, T, 2, H, W]
    occlusion: torch.Tensor       # [B, T, H, W]
    entity_count_logits: torch.Tensor  # [B, K]


class LearnedPromptPlanner(nn.Module):
    """A lightweight learned text-to-plan module.

    The planner predicts slot-wise trajectories, depth ordering, and visibility
    directly from text, then renders them into dense cues consumable by the
    anti-chimera scene hint builder.
    """

    def __init__(
        self,
        max_objects: int,
        num_frames: int,
        image_size: int,
        prompt_vocab_size: int = 4096,
        hidden_size: int = 256,
        max_prompt_tokens: int = 48,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_objects = int(max_objects)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.hidden_size = int(hidden_size)
        self.parser = PromptParser()
        self.prompt_encoder = SimplePromptEncoder(
            vocab_size=int(prompt_vocab_size),
            hidden_size=int(hidden_size),
            max_tokens=int(max_prompt_tokens),
            dropout=float(dropout),
        )
        self.slot_queries = nn.Parameter(torch.randn(self.max_objects, self.hidden_size) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(self.num_frames, self.hidden_size) * 0.02)
        self.count_embed = nn.Embedding(self.max_objects + 1, self.hidden_size)
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_size * 4,
            dropout=float(dropout),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.track_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 4),
        )
        self.depth_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.count_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.max_objects),
        )
        self.radius_min = 0.06
        self.radius_max = 0.22

    def _count_from_prompts(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        counts = []
        for prompt in prompts:
            parsed = self.parser.parse(prompt)
            counts.append(min(max(1, len(parsed.entities)), self.max_objects))
        return torch.tensor(counts, device=device, dtype=torch.long)

    def _slot_features(self, prompt_features: torch.Tensor, count_idx: torch.Tensor) -> torch.Tensor:
        B = prompt_features.shape[0]
        slot = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
        count_embed = self.count_embed(count_idx).unsqueeze(1)
        global_context = prompt_features.unsqueeze(1)
        slot = slot + global_context + count_embed
        slot = slot.unsqueeze(1).expand(B, self.num_frames, self.max_objects, self.hidden_size)
        time = self.time_embed.unsqueeze(0).unsqueeze(2).expand(B, self.num_frames, self.max_objects, self.hidden_size)
        fused = self.fusion(slot + time)
        fused = fused.reshape(B * self.max_objects, self.num_frames, self.hidden_size)
        fused = self.temporal_encoder(fused)
        return fused.reshape(B, self.num_frames, self.max_objects, self.hidden_size)

    def _decode_tracks(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.track_head(hidden)
        center = torch.sigmoid(raw[..., :2]) * 0.84 + 0.08
        radius = torch.sigmoid(raw[..., 2:]) * (self.radius_max - self.radius_min) + self.radius_min
        tracks = torch.cat([
            (center[..., :1] - radius[..., :1]).clamp(0.0, 1.0),
            (center[..., 1:2] - radius[..., 1:2]).clamp(0.0, 1.0),
            (center[..., :1] + radius[..., :1]).clamp(0.0, 1.0),
            (center[..., 1:2] + radius[..., 1:2]).clamp(0.0, 1.0),
        ], dim=-1)
        x1y1 = tracks[..., :2]
        x2y2 = torch.maximum(tracks[..., 2:], x1y1 + 0.02)
        return torch.cat([x1y1, x2y2.clamp(0.0, 1.0)], dim=-1)

    def _decode_depth(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.depth_head(hidden)[..., 0])

    def _decode_visibility(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.visibility_head(hidden)[..., 0]

    def _render_dense(self, tracks: torch.Tensor, depth_slots: torch.Tensor, visibility_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, K, _ = tracks.shape
        H = self.image_size
        W = self.image_size
        device = tracks.device
        ys = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, 1, H, 1)
        xs = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, 1, W)
        x1 = tracks[..., 0].unsqueeze(-1).unsqueeze(-1)
        y1 = tracks[..., 1].unsqueeze(-1).unsqueeze(-1)
        x2 = tracks[..., 2].unsqueeze(-1).unsqueeze(-1)
        y2 = tracks[..., 3].unsqueeze(-1).unsqueeze(-1)
        sharpness = 40.0
        mask_x = torch.sigmoid((xs - x1) * sharpness) * torch.sigmoid((x2 - xs) * sharpness)
        mask_y = torch.sigmoid((ys - y1) * sharpness) * torch.sigmoid((y2 - ys) * sharpness)
        masks = mask_x * mask_y
        visibility = torch.sigmoid(visibility_logits)
        masks = masks * visibility.unsqueeze(-1).unsqueeze(-1)
        occupancy = masks.sum(dim=2)
        occlusion = (occupancy > 1.0).float()
        weight = masks / masks.sum(dim=2, keepdim=True).clamp_min(1e-6)
        depth = (weight * depth_slots.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
        depth = torch.where(occupancy > 0, depth, torch.full_like(depth, 0.5))
        flow = torch.zeros(B, T, 2, H, W, device=device)
        centers = torch.stack([(tracks[..., 0] + tracks[..., 2]) * 0.5, (tracks[..., 1] + tracks[..., 3]) * 0.5], dim=-1)
        delta = centers[:, 1:] - centers[:, :-1]
        delta_x = delta[..., 0].unsqueeze(-1).unsqueeze(-1)
        delta_y = delta[..., 1].unsqueeze(-1).unsqueeze(-1)
        flow[:, :-1, 0] = (masks[:, :-1] * delta_x).sum(dim=2)
        flow[:, :-1, 1] = (masks[:, :-1] * delta_y).sum(dim=2)
        if T > 1:
            flow[:, -1] = flow[:, -2]
        return masks, depth, visibility, flow, occlusion

    def forward(self, prompts: list[str], device: torch.device | None = None) -> PlannerPrediction:
        if device is None:
            device = next(self.parameters()).device
        prompt_features = self.prompt_encoder(prompts, device=device)
        count_idx = self._count_from_prompts(prompts, device=device)
        hidden = self._slot_features(prompt_features, count_idx)
        tracks = self._decode_tracks(hidden)
        depth_slots = self._decode_depth(hidden)
        visibility_logits = self._decode_visibility(hidden)
        entity_count_logits = self.count_head(prompt_features)
        masks, depth, visibility, flow, occlusion = self._render_dense(tracks, depth_slots, visibility_logits)
        return PlannerPrediction(
            tracks=tracks,
            depth_slots=depth_slots,
            visibility_logits=visibility_logits,
            masks=masks,
            depth=depth,
            visibility=visibility,
            flow=flow,
            occlusion=occlusion,
            entity_count_logits=entity_count_logits,
        )

    @torch.no_grad()
    def plan(self, prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
        pred = self.forward([prompt], device=device)
        return {
            'tracks': pred.tracks[0],
            'depth': pred.depth[0],
            'visibility': pred.visibility[0],
            'masks': pred.masks[0],
            'flow': pred.flow[0],
            'occlusion': pred.occlusion[0],
        }


def planner_losses(
    pred: PlannerPrediction,
    gt_tracks: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_visibility: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_occlusion: torch.Tensor,
    gt_count: torch.Tensor,
    mask_weight: float = 1.0,
    occ_weight: float = 1.0,
    track_weight: float = 5.0,
    depth_weight: float = 2.0,
    visibility_weight: float = 1.0,
    count_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    track_loss = F.l1_loss(pred.tracks, gt_tracks)
    depth_loss = F.l1_loss(pred.depth, gt_depth)
    vis_loss = F.binary_cross_entropy_with_logits(pred.visibility_logits, gt_visibility)
    mask_loss = F.binary_cross_entropy(pred.masks.clamp(1e-5, 1 - 1e-5), gt_masks)
    occ_loss = F.binary_cross_entropy(pred.occlusion.clamp(1e-5, 1 - 1e-5), gt_occlusion)
    count_loss = F.cross_entropy(pred.entity_count_logits, gt_count.long() - 1)
    total = (
        track_weight * track_loss
        + depth_weight * depth_loss
        + visibility_weight * vis_loss
        + mask_weight * mask_loss
        + occ_weight * occ_loss
        + count_weight * count_loss
    )
    return {
        'total': total,
        'track': track_loss,
        'depth': depth_loss,
        'visibility': vis_loss,
        'mask': mask_loss,
        'occlusion': occ_loss,
        'count': count_loss,
    }


_PLANNER_CACHE: Dict[Tuple[str, str], LearnedPromptPlanner] = {}


def load_learned_planner(checkpoint_path: str | Path, config: Dict, device: torch.device) -> LearnedPromptPlanner:
    checkpoint_path = str(checkpoint_path)
    cache_key = (checkpoint_path, str(device))
    if cache_key in _PLANNER_CACHE:
        return _PLANNER_CACHE[cache_key]
    planner_cfg = dict(config.get('planner', {}))
    planner = LearnedPromptPlanner(
        max_objects=int(config['data']['max_objects']),
        num_frames=int(config['data']['num_frames']),
        image_size=int(config['data']['image_size']),
        prompt_vocab_size=int(planner_cfg.get('prompt_vocab_size', 4096)),
        hidden_size=int(planner_cfg.get('hidden_size', 256)),
        max_prompt_tokens=int(planner_cfg.get('max_prompt_tokens', 48)),
        num_layers=int(planner_cfg.get('num_layers', 4)),
        num_heads=int(planner_cfg.get('num_heads', 8)),
        dropout=float(planner_cfg.get('dropout', 0.1)),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['planner'] if isinstance(ckpt, dict) and 'planner' in ckpt else ckpt
    planner.load_state_dict(state, strict=False)
    planner.eval()
    _PLANNER_CACHE[cache_key] = planner
    return planner
