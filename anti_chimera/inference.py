from __future__ import annotations

from typing import Dict, List

import torch

from anti_chimera.data.scene_hint import SceneHintBuilder
from anti_chimera.text import PromptParser
from anti_chimera.trainer import DiffusionSchedule, make_schedule
from anti_chimera.utils import denormalize_video


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


def p_sample(model, x, t, prompts: List[str], cond, schedule: DiffusionSchedule):
    betas_t = schedule.betas[t][:, None, None, None, None]
    sqrt_one_minus = schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]
    sqrt_recip_alpha = (1.0 / torch.sqrt(schedule.alphas[t]))[:, None, None, None, None]
    pred_noise = model(x, t, prompts, cond)
    model_mean = sqrt_recip_alpha * (x - betas_t * pred_noise / sqrt_one_minus)
    if (t == 0).all():
        return model_mean
    noise = torch.randn_like(x)
    posterior_var = betas_t
    return model_mean + torch.sqrt(posterior_var) * noise


def sample_video(model, prompt: str, config: Dict, device: torch.device) -> torch.Tensor:
    model.eval()
    data_cfg = config['data']
    sampling_cfg = config['sampling']
    x = torch.randn(1, 3, data_cfg['num_frames'], data_cfg['image_size'], data_cfg['image_size'], device=device)
    cond = build_null_condition(prompt, config, device)
    schedule = make_schedule(sampling_cfg['num_steps'], device)
    for step in reversed(range(sampling_cfg['num_steps'])):
        t = torch.full((1,), step, device=device, dtype=torch.long)
        x = p_sample(model, x, t, [prompt], cond, schedule)
    return denormalize_video(x[0].detach().cpu())
