from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv3d(channels, channels, 3, stride=(1, 2, 2), padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        return self.conv(x)


class SceneInjector(nn.Module):
    def __init__(self, feat_ch: int, cond_ch: int, text_dim: int) -> None:
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Conv3d(cond_ch, feat_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(feat_ch, feat_ch, 3, padding=1),
        )
        self.text_proj = nn.Linear(text_dim, feat_ch)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        cond_feat = self.cond_proj(cond)
        text_bias = self.text_proj(text_emb)[:, :, None, None, None]
        return x + torch.tanh(self.gate) * (cond_feat + text_bias)
