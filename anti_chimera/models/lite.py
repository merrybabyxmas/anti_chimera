from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from anti_chimera.diffusion import SimpleDDPMScheduler
from anti_chimera.models.modules import Downsample3D, ResBlock3D, SceneInjector, Upsample3D, timestep_embedding
from anti_chimera.text import SimplePromptEncoder


def _pool_condition(cond: torch.Tensor, levels: int) -> List[torch.Tensor]:
    conds = [cond]
    for _ in range(1, levels):
        conds.append(F.avg_pool3d(conds[-1], kernel_size=(1, 2, 2), stride=(1, 2, 2)))
    return conds


class LiteVideoDenoiser(nn.Module):
    def __init__(
        self,
        cond_channels: int,
        base_channels: int = 32,
        prompt_vocab_size: int = 1024,
        prompt_dim: int = 128,
        num_train_timesteps: int = 1000,
        max_prompt_tokens: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError('base_channels should be divisible by 8 for GroupNorm stability')

        self.scheduler = SimpleDDPMScheduler(num_train_timesteps=num_train_timesteps)
        self.prompt_encoder = SimplePromptEncoder(
            vocab_size=prompt_vocab_size,
            hidden_size=prompt_dim,
            max_tokens=max_prompt_tokens,
            dropout=dropout,
        )

        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_proj = nn.Conv3d(3, base_channels, kernel_size=3, padding=1)

        self.inj0 = SceneInjector(base_channels, cond_channels, prompt_dim, gate_init=0.18)
        self.inj1 = SceneInjector(base_channels * 2, cond_channels, prompt_dim, gate_init=0.14)
        self.inj2 = SceneInjector(base_channels * 4, cond_channels, prompt_dim, gate_init=0.10)

        self.down0_a = ResBlock3D(base_channels, base_channels, time_dim, dropout=dropout)
        self.down0_b = ResBlock3D(base_channels, base_channels, time_dim, dropout=dropout)
        self.down0 = Downsample3D(base_channels)

        self.down1_a = ResBlock3D(base_channels, base_channels * 2, time_dim, dropout=dropout)
        self.down1_b = ResBlock3D(base_channels * 2, base_channels * 2, time_dim, dropout=dropout)
        self.down1 = Downsample3D(base_channels * 2)

        self.down2_a = ResBlock3D(base_channels * 2, base_channels * 4, time_dim, dropout=dropout)
        self.down2_b = ResBlock3D(base_channels * 4, base_channels * 4, time_dim, dropout=dropout)

        self.mid_a = ResBlock3D(base_channels * 4, base_channels * 4, time_dim, dropout=dropout)
        self.mid_b = ResBlock3D(base_channels * 4, base_channels * 4, time_dim, dropout=dropout)

        self.up2 = Upsample3D(base_channels * 4)
        self.up1_a = ResBlock3D(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim, dropout=dropout)
        self.up1_b = ResBlock3D(base_channels * 2, base_channels * 2, time_dim, dropout=dropout)

        self.up1 = Upsample3D(base_channels * 2)
        self.up0_a = ResBlock3D(base_channels * 2 + base_channels, base_channels, time_dim, dropout=dropout)
        self.up0_b = ResBlock3D(base_channels, base_channels, time_dim, dropout=dropout)

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_proj = nn.Conv3d(base_channels, 3, kernel_size=3, padding=1)

    @property
    def num_train_timesteps(self) -> int:
        return self.scheduler.num_train_timesteps

    @property
    def latent_scaling_factor(self) -> float:
        return 1.0

    def encode_prompts(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        return self.prompt_encoder(prompts, device=device)

    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        return videos

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.scheduler.add_noise(latents, noise, timesteps)

    def prediction_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return noise

    def infer_latent_shape(self, num_frames: int, height: int, width: int, device: torch.device) -> Tuple[int, int, int, int, int]:
        return (1, 3, num_frames, height, width)

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor, prompts: List[str], cond: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError(f'Expected latents with shape [B, C, T, H, W], got {tuple(latents.shape)}')
        t_emb = timestep_embedding(timesteps.long(), self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        text_emb = self.encode_prompts(prompts, device=latents.device)
        cond_levels = _pool_condition(cond, 3)

        x = self.input_proj(latents)
        x = self.inj0(x, cond_levels[0], text_emb)
        x = self.down0_a(x, t_emb)
        x = self.down0_b(x, t_emb)
        skip0 = x
        x = self.down0(x)

        x = self.down1_a(x, t_emb)
        x = self.inj1(x, cond_levels[1], text_emb)
        x = self.down1_b(x, t_emb)
        skip1 = x
        x = self.down1(x)

        x = self.down2_a(x, t_emb)
        x = self.inj2(x, cond_levels[2], text_emb)
        x = self.down2_b(x, t_emb)

        x = self.mid_a(x, t_emb)
        x = self.mid_b(x, t_emb)

        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up1_a(x, t_emb)
        x = self.inj1(x, cond_levels[1], text_emb)
        x = self.up1_b(x, t_emb)

        x = self.up1(x)
        x = torch.cat([x, skip0], dim=1)
        x = self.up0_a(x, t_emb)
        x = self.inj0(x, cond_levels[0], text_emb)
        x = self.up0_b(x, t_emb)

        x = self.out_proj(F.silu(self.out_norm(x)))
        return x
