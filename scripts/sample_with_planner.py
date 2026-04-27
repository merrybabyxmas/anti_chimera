from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anti_chimera.config import load_config
from anti_chimera.inference import sample_video
from anti_chimera.models.model import AntiChimeraVideoDiffusion
from anti_chimera.utils import default_device, save_gif, save_video_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--planner-checkpoint', required=True, type=str)
    parser.add_argument('--prompt', required=True, type=str)
    parser.add_argument('--out', required=True, type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault('planner', {})['checkpoint'] = args.planner_checkpoint
    device = default_device(config.get('training', {}).get('device', 'cuda'))
    builder_channels = int(config['data']['max_objects']) * 4 + int(config['data']['depth_bins']) + 4
    try:
        from anti_chimera.data.scene_hint import SceneHintBuilder
        builder_channels = SceneHintBuilder(
            max_objects=int(config['data']['max_objects']),
            depth_bins=int(config['data']['depth_bins']),
            image_size=int(config['data']['image_size']),
        ).num_channels()
    except Exception:
        pass
    model = AntiChimeraVideoDiffusion(cond_channels=builder_channels, **config['model']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    video = sample_video(model, args.prompt, config, device, cond=None)
    save_gif(video * 2 - 1, args.out)
    png_out = str(Path(args.out).with_suffix('.png'))
    save_video_png(video * 2 - 1, png_out)


if __name__ == '__main__':
    main()
