from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


COGVIDEOX_VARIANTS = {
    '2b': 'CogVideoX-2B compatible',
    '5b': 'CogVideoX-5B compatible',
}


class AntiChimeraVideoDiffusion(nn.Module):
    """CogVideoX wrapper that keeps the pretrained pipeline components and only adds anti-chimera conditioning.

    This wrapper is intentionally reuse-first:
    - transformer/text encoder/tokenizer/vae/scheduler come from a CogVideoX pipeline
    - the only custom parts are scene-hint projection and gated conditioning injection
    """

    def __init__(
        self,
        cond_channels: int,
        pretrained_model_name_or_path: Optional[str],
        variant: str = '2b',
        transformer_subfolder: str = 'transformer',
        tokenizer_subfolder: Optional[str] = None,
        text_encoder_subfolder: Optional[str] = None,
        vae_subfolder: Optional[str] = None,
        scheduler_subfolder: Optional[str] = None,
        freeze_text_encoder: bool = True,
        freeze_vae: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        if variant not in COGVIDEOX_VARIANTS:
            raise ValueError(f'Unsupported CogVideoX variant: {variant}. Expected one of {list(COGVIDEOX_VARIANTS)}')
        if not pretrained_model_name_or_path:
            raise ValueError(
                'CogVideoX mode is reuse-first and expects a pretrained_model_name_or_path for either 2B or 5B.'
            )

        try:
            from diffusers import CogVideoXPipeline
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                'CogVideoXPipeline is not available in the installed diffusers version. '
                'Please install a diffusers release with CogVideoX support.'
            ) from exc

        pipe = CogVideoXPipeline.from_pretrained(pretrained_model_name_or_path)
        self.pipe = pipe
        self.variant = variant
        self.transformer = pipe.transformer
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler

        if freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
        if freeze_vae:
            self.vae.requires_grad_(False)
            self.vae.eval()

        latent_channels = int(getattr(self.transformer.config, 'in_channels', 16))
        text_hidden = int(
            getattr(self.text_encoder.config, 'hidden_size', None)
            or getattr(self.text_encoder.config, 'd_model', None)
            or 4096
        )

        self.cond_to_latent = nn.Sequential(
            nn.Conv3d(cond_channels, latent_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(latent_channels, latent_channels, kernel_size=3, padding=1),
        )
        self.cond_pool = nn.AdaptiveAvgPool3d((4, 1, 1))
        self.cond_token_proj = nn.Linear(cond_channels, text_hidden)
        self.input_gate = nn.Parameter(torch.tensor(0.0))
        self.token_gate = nn.Parameter(torch.tensor(0.0))

    @property
    def num_train_timesteps(self) -> int:
        return int(getattr(self.scheduler.config, 'num_train_timesteps', 1000))

    @property
    def latent_scaling_factor(self) -> float:
        return float(getattr(self.vae.config, 'scaling_factor', 1.0))

    def encode_prompts(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=min(getattr(self.tokenizer, 'model_max_length', 226), 226),
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = self.text_encoder(**encoded)
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        return outputs[0]

    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        # videos expected in [-1, 1], [B, C, T, H, W]
        posterior = self.vae.encode(videos)
        latent_dist = posterior.latent_dist if hasattr(posterior, 'latent_dist') else posterior
        latents = latent_dist.sample()
        return latents * self.latent_scaling_factor

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.latent_scaling_factor
        decoded = self.vae.decode(latents)
        return decoded.sample if hasattr(decoded, 'sample') else decoded[0]

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.scheduler.add_noise(latents, noise, timesteps)

    def prediction_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        prediction_type = getattr(self.scheduler.config, 'prediction_type', 'epsilon')
        if prediction_type == 'v_prediction' and hasattr(self.scheduler, 'get_velocity'):
            return self.scheduler.get_velocity(latents, noise, timesteps)
        return noise

    def build_condition(self, cond: torch.Tensor, target_shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        cond_latent = F.interpolate(cond, size=target_shape, mode='trilinear', align_corners=False)
        cond_latent = self.cond_to_latent(cond_latent)
        cond_tokens = self.cond_pool(cond).flatten(start_dim=2).transpose(1, 2)
        cond_tokens = self.cond_token_proj(cond_tokens)
        return cond_latent, cond_tokens

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor, prompts: List[str], cond: torch.Tensor) -> torch.Tensor:
        device = latents.device
        text_hidden = self.encode_prompts(prompts, device=device)
        cond_latent, cond_tokens = self.build_condition(cond, latents.shape[-3:])
        hidden_states = latents + torch.tanh(self.input_gate) * cond_latent
        encoder_hidden_states = torch.cat([text_hidden, torch.tanh(self.token_gate) * cond_tokens], dim=1)
        out = self.transformer(
            hidden_states=hidden_states,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        return out[0]

    @torch.no_grad()
    def infer_latent_shape(self, num_frames: int, height: int, width: int, device: torch.device) -> Tuple[int, int, int, int, int]:
        dummy = torch.zeros(1, 3, num_frames, height, width, device=device)
        latents = self.encode_video(dummy)
        return tuple(latents.shape)
