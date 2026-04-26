from __future__ import annotations

import argparse
from pathlib import Path

import torch

from anti_chimera.config import load_config
from anti_chimera.inference import sample_video
from anti_chimera.models.model import AntiChimeraVideoDiffusion
from anti_chimera.utils import default_device, save_gif, save_video_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = default_device(config['training']['device'])
    data_cfg = config['data']
    model_cfg = config['model']
    cond_channels = data_cfg['max_objects'] + data_cfg['depth_bins'] + data_cfg['max_objects']
    model = AntiChimeraVideoDiffusion(cond_channels=cond_channels, **model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = dict(ckpt.get('ema_model') or ckpt['model'])
    for key in list(state.keys()):
        if key.endswith('scheduler.timesteps'):
            state.pop(key)
    model.load_state_dict(state, strict=False)
    video = sample_video(model, args.prompt, config, device)
    video_path = Path(args.out)
    save_gif(video * 2 - 1, video_path)
    save_video_png(video * 2 - 1, video_path.with_suffix('.png'))


if __name__ == '__main__':
    main()
