from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentControlEncoder(nn.Module):
    """Project scene sidecars into a latent residual for direct diffusion control."""

    def __init__(
        self,
        cond_channels: int,
        latent_channels: int,
        hidden_channels: int = 256,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(cond_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, latent_channels, kernel_size=1),
        )
        nn.init.zeros_(self.proj[-1].weight)
        if self.proj[-1].bias is not None:
            nn.init.zeros_(self.proj[-1].bias)
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, cond: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        if cond.ndim != 5:
            raise ValueError(f'expected cond tensor with 5 dims, got {tuple(cond.shape)}')
        pooled = F.adaptive_avg_pool3d(cond.float(), target_shape).to(dtype=self.proj[0].weight.dtype)
        residual = self.proj(pooled)
        residual = residual.permute(0, 2, 1, 3, 4).contiguous()
        return torch.sigmoid(self.gate) * residual


class ReferenceConditionEncoder(nn.Module):
    """Encode a reference-frame latent into a text-space bias."""

    def __init__(
        self,
        latent_channels: int,
        hidden_size: int,
        hidden_channels: int = 256,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Sequential(
            nn.LayerNorm(latent_channels),
            nn.Linear(latent_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_size, bias=False),
        )
        nn.init.zeros_(self.proj[-1].weight)
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, reference_latent: torch.Tensor) -> torch.Tensor:
        if reference_latent.ndim != 5:
            raise ValueError(f'expected reference_latent tensor with 5 dims, got {tuple(reference_latent.shape)}')
        pooled = self.pool(reference_latent.float()).flatten(1).to(dtype=self.gate.dtype)
        bias = self.proj(pooled)
        return (1.0 + torch.sigmoid(self.gate)) * bias
