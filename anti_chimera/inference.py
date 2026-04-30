from __future__ import annotations

from typing import Dict

import torch

from anti_chimera.data.scene_hint_modes import build_scene_hint_builder
from anti_chimera.planner import PromptScenePlanner
from anti_chimera.text import PromptParser


def build_null_condition(prompt: str, config: Dict, device: torch.device) -> torch.Tensor:
    data_cfg = config['data']
    T = int(data_cfg['num_frames'])
    H = int(data_cfg['image_size'])
    max_objects = int(data_cfg['max_objects'])
    parser = PromptParser()
    parsed = parser.parse(prompt)
    planner = PromptScenePlanner(max_objects=max_objects, num_frames=T, image_size=H)
    planned = planner.plan(prompt, device=device)
    if len(parsed.entities) == 0:
        sample = {
            'video': torch.zeros(3, T, H, H, device=device),
            'caption': prompt,
            'tracks': torch.zeros(T, max_objects, 4, device=device),
            'depth': torch.full((T, H, H), 0.5, device=device),
            'visibility': torch.zeros(T, max_objects, device=device),
            'masks': torch.zeros(T, max_objects, H, H, device=device),
            'flow': torch.zeros(T, 2, H, H, device=device),
            'occlusion': torch.zeros(T, H, H, device=device),
        }
    else:
        sample = {
            'video': torch.zeros(3, T, H, H, device=device),
            'caption': prompt,
            'tracks': planned.tracks,
            'depth': planned.depth,
            'visibility': planned.visibility,
            'masks': planned.masks,
            'flow': planned.flow,
            'occlusion': planned.occlusion,
        }
    builder = build_scene_hint_builder(data_cfg)
    cond = builder.build(sample).unsqueeze(0).to(device)
    if not prompt.strip():
        cond.zero_()
    return cond


@torch.no_grad()
def sample_video(
    model,
    prompt: str,
    config: Dict,
    device: torch.device,
    cond: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    data_cfg = config['data']
    sampling_cfg = config['sampling']
    if cond is None:
        cond = build_null_condition(prompt, config, device)
    else:
        if cond.ndim == 4:
            cond = cond.unsqueeze(0)
        cond = cond.to(device)
    guidance_scale = float(sampling_cfg.get('guidance_scale', 1.0))
    use_cfg = guidance_scale > 1.0

    num_steps = int(sampling_cfg['num_steps'])
    if hasattr(model.scheduler, 'set_timesteps'):
        model.scheduler.set_timesteps(num_steps, device=device)
        timesteps = model.scheduler.timesteps
    else:
        timesteps = torch.arange(num_steps - 1, -1, -1, device=device)

    latent_shape = model.infer_latent_shape(
        num_frames=int(data_cfg['num_frames']),
        height=int(data_cfg['image_size']),
        width=int(data_cfg['image_size']),
        device=device,
    )
    latent_dtype = torch.float16 if device.type == 'cuda' else next(model.parameters()).dtype
    latents = torch.randn(latent_shape, device=device, dtype=latent_dtype)

    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=torch.float16):
        for timestep in timesteps:
            t = timestep if torch.is_tensor(timestep) else torch.tensor(timestep, device=device)
            t_batch = t.reshape(1).long()
            if use_cfg:
                uncond = model(latents, t_batch, [''], torch.zeros_like(cond))
                cond_pred = model(latents, t_batch, [prompt], cond)
                noise_pred = uncond + guidance_scale * (cond_pred - uncond)
            else:
                noise_pred = model(latents, t_batch, [prompt], cond)
            step_out = model.scheduler.step(noise_pred, t, latents)
            latents = step_out.prev_sample if hasattr(step_out, 'prev_sample') else step_out[0]

    video = model.decode_latents(latents)
    video = ((video[0].detach().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
    return video
