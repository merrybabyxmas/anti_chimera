from __future__ import annotations

from typing import Dict

import torch

from anti_chimera.data.scene_hint import SceneHintBuilder
from anti_chimera.text import PromptParser


def build_null_condition(prompt: str, config: Dict, device: torch.device) -> torch.Tensor:
    data_cfg = config['data']
    T = data_cfg['num_frames']
    H = data_cfg['image_size']
    max_objects = data_cfg['max_objects']
    tracks = torch.zeros(T, max_objects, 4)
    visibility = torch.zeros(T, max_objects)
    parser = PromptParser()
    parsed = parser.parse(prompt)
    n = min(len(parsed.entities), max_objects)
    centers = [(0.25, 0.5), (0.75, 0.5), (0.5, 0.25)]
    targets = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.75)]
    r = 0.12
    for k in range(n):
        for t in range(T):
            alpha = t / max(T - 1, 1)
            cx = centers[k][0] * (1 - alpha) + targets[k][0] * alpha
            cy = centers[k][1] * (1 - alpha) + targets[k][1] * alpha
            tracks[t, k] = torch.tensor([max(0.0, cx - r), max(0.0, cy - r), min(1.0, cx + r), min(1.0, cy + r)])
            visibility[t, k] = 1.0
    sample = {
        'video': torch.zeros(3, T, H, H),
        'caption': prompt,
        'tracks': tracks,
        'depth': torch.full((T, H, H), 0.5),
        'visibility': visibility,
    }
    builder = SceneHintBuilder(max_objects=max_objects, depth_bins=data_cfg['depth_bins'], image_size=H)
    cond = builder.build(sample).unsqueeze(0).to(device)
    return cond


@torch.no_grad()
def sample_video(model, prompt: str, config: Dict, device: torch.device) -> torch.Tensor:
    model.eval()
    data_cfg = config['data']
    sampling_cfg = config['sampling']
    cond = build_null_condition(prompt, config, device)

    num_steps = int(sampling_cfg['num_steps'])
    if hasattr(model.scheduler, 'set_timesteps'):
        model.scheduler.set_timesteps(num_steps, device=device)
        timesteps = model.scheduler.timesteps
    else:
        timesteps = torch.arange(num_steps - 1, -1, -1, device=device)

    latent_shape = model.infer_latent_shape(
        num_frames=data_cfg['num_frames'],
        height=data_cfg['image_size'],
        width=data_cfg['image_size'],
        device=device,
    )
    latents = torch.randn(latent_shape, device=device, dtype=next(model.parameters()).dtype)

    for timestep in timesteps:
        t = timestep if torch.is_tensor(timestep) else torch.tensor(timestep, device=device)
        t_batch = t.reshape(1).long()
        noise_pred = model(latents, t_batch, [prompt], cond)
        step_out = model.scheduler.step(noise_pred, t, latents)
        latents = step_out.prev_sample if hasattr(step_out, 'prev_sample') else step_out[0]

    video = model.decode_latents(latents)
    video = ((video[0].detach().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
    return video
