from __future__ import annotations

from typing import Dict

import torch

from anti_chimera.data.scene_hint import SceneHintBuilder
from anti_chimera.planner_learned import load_learned_planner


def build_planned_condition(prompt: str, config: Dict, device: torch.device, planner_checkpoint: str) -> torch.Tensor:
    planner = load_learned_planner(planner_checkpoint, config, device=device)
    planned = planner.plan(prompt, device=device)
    data_cfg = config['data']
    T = int(data_cfg['num_frames'])
    H = int(data_cfg['image_size'])
    sample = {
        'video': torch.zeros(3, T, H, H, device=device),
        'caption': prompt,
        'tracks': planned['tracks'],
        'depth': planned['depth'],
        'visibility': planned['visibility'],
        'masks': planned['masks'],
        'flow': planned['flow'],
        'occlusion': planned['occlusion'],
    }
    builder = SceneHintBuilder(
        max_objects=int(data_cfg['max_objects']),
        depth_bins=int(data_cfg['depth_bins']),
        image_size=H,
    )
    return builder.build(sample).unsqueeze(0).to(device)


@torch.no_grad()
def sample_video_with_planner(
    model,
    prompt: str,
    config: Dict,
    device: torch.device,
    planner_checkpoint: str,
) -> torch.Tensor:
    from anti_chimera.inference import sample_video

    cond = build_planned_condition(prompt, config, device, planner_checkpoint)
    return sample_video(model, prompt, config, device, cond=cond)
