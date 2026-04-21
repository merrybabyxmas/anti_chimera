from __future__ import annotations

import argparse

import torch

from anti_chimera.config import load_config
from anti_chimera.inference import sample_video
from anti_chimera.models.model import AntiChimeraVideoDiffusion
from anti_chimera.utils import default_device, save_gif


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
    model = AntiChimeraVideoDiffusion(
        in_channels=3,
        cond_channels=cond_channels,
        vocab_size=data_cfg['vocab_size'],
        base_channels=model_cfg['base_channels'],
        channel_multipliers=model_cfg['channel_multipliers'],
        time_embed_dim=model_cfg['time_embed_dim'],
        text_embed_dim=model_cfg['text_embed_dim'],
        dropout=model_cfg['dropout'],
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    video = sample_video(model, args.prompt, config, device)
    save_gif(video * 2 - 1, args.out)


if __name__ == '__main__':
    main()
