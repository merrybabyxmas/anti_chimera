from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from anti_chimera.models.lite import LiteVideoDenoiser


COGVIDEOX_VARIANTS = {
    '2b': 'CogVideoX-2B compatible',
    '5b': 'CogVideoX-5B compatible',
}


def _logit(prob: float) -> float:
    prob = min(max(prob, 1e-4), 1.0 - 1e-4)
    return math.log(prob / (1.0 - prob))


class _CogVideoXImpl(nn.Module):
    """CogVideoX wrapper that keeps pretrained components and adds anti-chimera conditioning."""

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
        freeze_transformer: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        **_: object,
    ) -> None:
        super().__init__()
        if variant not in COGVIDEOX_VARIANTS:
            raise ValueError(f'Unsupported CogVideoX variant: {variant}. Expected one of {list(COGVIDEOX_VARIANTS)}')
        if not pretrained_model_name_or_path:
            raise ValueError('CogVideoX mode expects pretrained_model_name_or_path.')

        try:
            from diffusers import CogVideoXPipeline
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                'CogVideoXPipeline is not available in the installed diffusers version. '
                'Install diffusers with CogVideoX support or use backend=lite3d.'
            ) from exc

        pipe = CogVideoXPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            transformer_subfolder=transformer_subfolder,
            tokenizer_subfolder=tokenizer_subfolder,
            text_encoder_subfolder=text_encoder_subfolder,
            vae_subfolder=vae_subfolder,
            scheduler_subfolder=scheduler_subfolder,
        )
        self.pipe = pipe
        self.variant = variant
        self.transformer = pipe.transformer
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler

        if freeze_transformer:
            self.transformer.requires_grad_(False)
            self.transformer.eval()
        if freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
        if freeze_vae:
            self.vae.requires_grad_(False)
            self.vae.eval()

        latent_channels = int(getattr(self.transformer.config, "in_channels", 16))
        text_hidden = int(
            getattr(self.text_encoder.config, "hidden_size", None)
            or getattr(self.text_encoder.config, "d_model", None)
            or 4096
        )

        vae_latent_channels = int(getattr(self.vae.config, "latent_channels", 4))
        self.latent_in_proj = nn.Conv3d(vae_latent_channels, latent_channels, kernel_size=1)
        self.latent_out_proj = nn.Conv3d(latent_channels, vae_latent_channels, kernel_size=1)

        self.cond_to_latent = nn.Sequential(
            nn.Conv3d(cond_channels, latent_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(latent_channels, latent_channels, kernel_size=3, padding=1),
        )
        self.cond_pool = nn.AdaptiveAvgPool3d((4, 1, 1))
        self.cond_token_proj = nn.Linear(cond_channels, text_hidden)
        self.input_gate = nn.Parameter(torch.tensor(_logit(0.15)))
        self.token_gate = nn.Parameter(torch.tensor(_logit(0.10)))
        self.conditioning_scale = 1.5

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
        vae_param = next(self.vae.parameters())
        videos = videos.to(device=vae_param.device, dtype=vae_param.dtype)
        posterior = self.vae.encode(videos)
        latent_dist = posterior.latent_dist if hasattr(posterior, 'latent_dist') else posterior
        latents = latent_dist.sample()
        return latents * self.latent_scaling_factor

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_param = next(self.vae.parameters())
        latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
        latents = latents / self.latent_scaling_factor
        decoded = self.vae.decode(latents)
        return decoded.sample if hasattr(decoded, 'sample') else decoded[0]

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.scheduler.add_noise(latents, noise, timesteps)

    def prediction_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        prediction_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "v_prediction" and hasattr(self.scheduler, "get_velocity"):
            return self.scheduler.get_velocity(latents, noise, timesteps)
        return noise

    def build_condition(self, cond: torch.Tensor, target_shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        cond_latent = torch.nn.functional.interpolate(cond, size=target_shape, mode="trilinear", align_corners=False)
        cond_latent = self.cond_to_latent(cond_latent)
        cond_tokens = self.cond_pool(cond).flatten(start_dim=2).transpose(1, 2)
        cond_tokens = self.cond_token_proj(cond_tokens)
        return cond_latent, cond_tokens

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor, prompts: List[str], cond: torch.Tensor) -> torch.Tensor:
        text_hidden = self.encode_prompts(prompts, device=latents.device)
        latents = self.latent_in_proj(latents)
        cond_latent, cond_tokens = self.build_condition(cond, latents.shape[-3:])
        hidden_states = latents + self.conditioning_scale * torch.sigmoid(self.input_gate) * cond_latent
        encoder_hidden_states = torch.cat(
            [text_hidden, self.conditioning_scale * torch.sigmoid(self.token_gate) * cond_tokens],
            dim=1,
        )
        out = self.transformer(
            hidden_states=hidden_states,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        return self.latent_out_proj(out[0])

    @torch.no_grad()
    def infer_latent_shape(self, num_frames: int, height: int, width: int, device: torch.device) -> Tuple[int, int, int, int, int]:
        dummy = torch.zeros(1, 3, num_frames, height, width, device=device)
        latents = self.encode_video(dummy)
        return tuple(latents.shape)



class AntiChimeraVideoDiffusion(nn.Module):
    """Backend dispatcher for the anti-chimera experiments.

    - backend=cogvideox: reuse a pretrained CogVideoX pipeline
    - backend=lite3d: fully self-contained video diffusion backbone for quick, GPU-friendly experiments
    """

    def __init__(
        self,
        cond_channels: int,
        backend: str = "lite3d",
        pretrained_model_name_or_path: Optional[str] = None,
        variant: str = "2b",
        transformer_subfolder: str = "transformer",
        tokenizer_subfolder: Optional[str] = None,
        text_encoder_subfolder: Optional[str] = None,
        vae_subfolder: Optional[str] = None,
        scheduler_subfolder: Optional[str] = None,
        freeze_text_encoder: bool = True,
        freeze_vae: bool = True,
        freeze_transformer: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        base_channels: int = 32,
        prompt_vocab_size: int = 1024,
        prompt_dim: int = 128,
        num_train_timesteps: int = 1000,
        max_prompt_tokens: int = 32,
        dropout: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.backend = backend
        if backend == "cogvideox":
            self.impl = _CogVideoXImpl(
                cond_channels=cond_channels,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                variant=variant,
                transformer_subfolder=transformer_subfolder,
                tokenizer_subfolder=tokenizer_subfolder,
                text_encoder_subfolder=text_encoder_subfolder,
                vae_subfolder=vae_subfolder,
                scheduler_subfolder=scheduler_subfolder,
                freeze_text_encoder=freeze_text_encoder,
                freeze_vae=freeze_vae,
                freeze_transformer=freeze_transformer,
                torch_dtype=torch_dtype,
                **kwargs,
            )
        elif backend == "lite3d":
            self.impl = LiteVideoDenoiser(
                cond_channels=cond_channels,
                base_channels=base_channels,
                prompt_vocab_size=prompt_vocab_size,
                prompt_dim=prompt_dim,
                num_train_timesteps=num_train_timesteps,
                max_prompt_tokens=max_prompt_tokens,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @property
    def scheduler(self):
        return self.impl.scheduler

    @property
    def num_train_timesteps(self) -> int:
        return int(self.impl.num_train_timesteps)

    @property
    def latent_scaling_factor(self) -> float:
        return float(self.impl.latent_scaling_factor)

    def encode_prompts(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        return self.impl.encode_prompts(prompts, device)

    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        return self.impl.encode_video(videos)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.impl.decode_latents(latents)

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.impl.add_noise(latents, noise, timesteps)

    def prediction_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.impl.prediction_target(latents, noise, timesteps)

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor, prompts: List[str], cond: torch.Tensor) -> torch.Tensor:
        return self.impl(latents, timesteps, prompts, cond)

    @torch.no_grad()
    def infer_latent_shape(self, num_frames: int, height: int, width: int, device: torch.device) -> tuple[int, int, int, int, int]:
        return self.impl.infer_latent_shape(num_frames, height, width, device)
