from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from anti_chimera.models.modules import (
    Downsample3D,
    ResBlock3D,
    SceneInjector,
    Upsample3D,
    timestep_embedding,
)
from anti_chimera.text import TokenTextEncoder


class AntiChimeraVideoDiffusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        vocab_size: int,
        base_channels: int = 64,
        channel_multipliers: List[int] | None = None,
        time_embed_dim: int = 256,
        text_embed_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channel_multipliers = channel_multipliers or [1, 2, 4]
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        self.text_encoder = TokenTextEncoder(vocab_size=vocab_size, embed_dim=text_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        self.cond_adapter = nn.Conv3d(cond_channels, base_channels, 3, padding=1)

        chs = [base_channels * m for m in channel_multipliers]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.injectors = nn.ModuleList()
        cur = base_channels
        for i, ch in enumerate(chs):
            self.down_blocks.append(ResBlock3D(cur, ch, time_embed_dim, dropout=dropout))
            self.injectors.append(SceneInjector(ch, base_channels, text_embed_dim))
            cur = ch
            if i != len(chs) - 1:
                self.downsamples.append(Downsample3D(cur))

        self.mid1 = ResBlock3D(cur, cur, time_embed_dim, dropout=dropout)
        self.mid2 = ResBlock3D(cur, cur, time_embed_dim, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in enumerate(reversed(chs)):
            self.up_blocks.append(ResBlock3D(cur + ch, ch, time_embed_dim, dropout=dropout))
            cur = ch
            if i != len(chs) - 1:
                self.upsamples.append(Upsample3D(cur))

        self.out_norm = nn.GroupNorm(8, cur)
        self.out_conv = nn.Conv3d(cur, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, prompts: List[str], cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(timesteps, self.time_embed_dim))
        text_emb = self.text_encoder(prompts)
        h = self.input_conv(x)
        cond_feats = self.cond_adapter(cond)

        skips = []
        ds_idx = 0
        for i, block in enumerate(self.down_blocks):
            if cond_feats.shape[-1] != h.shape[-1] or cond_feats.shape[-2] != h.shape[-2]:
                cond_feats = F.interpolate(cond_feats, size=h.shape[-3:], mode='trilinear', align_corners=False)
            h = block(h, t_emb)
            h = self.injectors[i](h, cond_feats, text_emb)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[ds_idx](h)
                ds_idx += 1

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        us_idx = 0
        for i, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if skip.shape[-3:] != h.shape[-3:]:
                h = self.upsamples[us_idx](h)
                us_idx += 1
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))
