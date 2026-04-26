from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from anti_chimera.utils import cosine_beta_schedule


@dataclass
class DDIMStepOutput:
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


class SimpleDDPMScheduler(nn.Module):
    """Minimal DDIM-style scheduler for the lightweight backend.

    The implementation is intentionally small but fully functional:
    - cosine beta schedule for training
    - epsilon prediction target
    - deterministic reverse updates during sampling
    """

    def __init__(self, num_train_timesteps: int = 1000) -> None:
        super().__init__()
        self.config = type('Config', (), {'num_train_timesteps': num_train_timesteps, 'prediction_type': 'epsilon'})()
        self.register_buffer('betas', cosine_beta_schedule(num_train_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', torch.cat([torch.ones(1), torch.cumprod(self.alphas, dim=0)[:-1]], dim=0))
        self.register_buffer('timesteps', torch.arange(num_train_timesteps - 1, -1, -1, dtype=torch.long), persistent=False)

    @property
    def num_train_timesteps(self) -> int:
        return int(self.config.num_train_timesteps)

    def set_timesteps(self, num_inference_steps: int, device: torch.device | None = None) -> None:
        # `torch.linspace` does not reliably support integer dtypes across torch versions.
        # Generate in float and quantize to the discrete scheduler grid.
        timesteps = torch.linspace(
            float(self.num_train_timesteps - 1),
            0.0,
            num_inference_steps,
            dtype=torch.float32,
            device=device,
        ).round().long()
        timesteps = torch.unique_consecutive(timesteps)
        if timesteps.numel() == 0:
            timesteps = torch.tensor([self.num_train_timesteps - 1], device=device, dtype=torch.long)
        elif timesteps[0] != self.num_train_timesteps - 1:
            timesteps = torch.cat([torch.tensor([self.num_train_timesteps - 1], device=timesteps.device, dtype=torch.long), timesteps])
        self.timesteps = timesteps

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        out = values.gather(0, timesteps.long())
        while out.ndim < sample.ndim:
            out = out[..., None]
        return out

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alphas_cumprod = self._extract(self.alphas_cumprod, timesteps, original_samples)
        return alphas_cumprod.sqrt() * original_samples + (1.0 - alphas_cumprod).sqrt() * noise

    def get_velocity(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alphas_cumprod = self._extract(self.alphas_cumprod, timesteps, original_samples)
        return alphas_cumprod.sqrt() * noise - (1.0 - alphas_cumprod).sqrt() * original_samples

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor | int, sample: torch.Tensor) -> DDIMStepOutput:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=sample.device, dtype=torch.long)
        timestep = timestep.long()
        alphas_cumprod_t = self._extract(self.alphas_cumprod, timestep, sample)

        if self.timesteps.ndim == 0:
            prev_timestep = torch.tensor(0, device=sample.device, dtype=torch.long)
        else:
            matches = (self.timesteps == timestep).nonzero(as_tuple=False)
            if len(matches) == 0:
                index = 0
            else:
                index = int(matches[0].item())
            if index + 1 < len(self.timesteps):
                prev_timestep = self.timesteps[index + 1].to(sample.device)
            else:
                prev_timestep = torch.tensor(0, device=sample.device, dtype=torch.long)

        # For arbitrary inference timesteps, the previous alpha comes from the
        # actual next scheduler index. Only the terminal step uses alpha_prev=1.
        if int(prev_timestep.item()) == 0:
            alphas_cumprod_prev = torch.ones_like(alphas_cumprod_t)
        else:
            alphas_cumprod_prev = self._extract(self.alphas_cumprod, prev_timestep, sample)
        pred_original_sample = (sample - (1.0 - alphas_cumprod_t).sqrt() * model_output) / alphas_cumprod_t.sqrt().clamp_min(1e-8)
        pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)
        prev_sample = alphas_cumprod_prev.sqrt() * pred_original_sample + (1.0 - alphas_cumprod_prev).sqrt() * model_output
        return DDIMStepOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
