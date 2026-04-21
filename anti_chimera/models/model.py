from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from diffusers import UNet3DConditionModel

from anti_chimera.text import HFTextEncoder


class AntiChimeraVideoDiffusion(nn.Module):
    """Wrap a diffusers UNet3DConditionModel instead of training a backbone from scratch."""

    def __init__(
        self,
        cond_channels: int,
        text_encoder_name_or_path: str,
        pretrained_model_name_or_path: Optional[str] = None,
        unet_subfolder: str = 'unet',
        freeze_text_encoder: bool = True,
        in_channels: int = 3,
        cross_attention_dim: int = 512,
        base_channels: int = 320,
        attention_head_dim: int = 8,
        layers_per_block: int = 1,
        **_: object,
    ) -> None:
        super().__init__()
        self.text_encoder = HFTextEncoder(text_encoder_name_or_path, freeze=freeze_text_encoder)
        text_hidden = self.text_encoder.hidden_size

        if pretrained_model_name_or_path:
            self.unet = UNet3DConditionModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder=unet_subfolder,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
            cross_attention_dim = int(self.unet.config.cross_attention_dim)
            in_channels = int(self.unet.config.in_channels)
        else:
            self.unet = UNet3DConditionModel(
                sample_size=None,
                in_channels=in_channels,
                out_channels=in_channels,
                down_block_types=(
                    'CrossAttnDownBlock3D',
                    'CrossAttnDownBlock3D',
                    'DownBlock3D',
                ),
                up_block_types=(
                    'UpBlock3D',
                    'CrossAttnUpBlock3D',
                    'CrossAttnUpBlock3D',
                ),
                block_out_channels=(base_channels, base_channels * 2, base_channels * 4),
                layers_per_block=layers_per_block,
                cross_attention_dim=cross_attention_dim,
                attention_head_dim=attention_head_dim,
            )

        self.cond_to_input = nn.Sequential(
            nn.Conv3d(cond_channels, in_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        self.cond_to_tokens = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 1, 1)),
            nn.Flatten(start_dim=2),
        )
        self.cond_token_proj = nn.Linear(cond_channels, cross_attention_dim)
        self.text_proj = nn.Linear(text_hidden, cross_attention_dim) if text_hidden != cross_attention_dim else nn.Identity()
        self.input_gate = nn.Parameter(torch.tensor(0.0))
        self.token_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, prompts: List[str], cond: torch.Tensor) -> torch.Tensor:
        device = x.device
        text_hidden = self.text_encoder(prompts, device=device)
        text_hidden = self.text_proj(text_hidden)

        cond_input = self.cond_to_input(cond)
        x_in = x + torch.tanh(self.input_gate) * cond_input

        cond_tokens = self.cond_to_tokens(cond).transpose(1, 2)
        cond_tokens = self.cond_token_proj(cond_tokens)
        encoder_hidden_states = torch.cat([text_hidden, torch.tanh(self.token_gate) * cond_tokens], dim=1)

        out = self.unet(x_in, timesteps, encoder_hidden_states=encoder_hidden_states)
        return out.sample if hasattr(out, 'sample') else out[0]
