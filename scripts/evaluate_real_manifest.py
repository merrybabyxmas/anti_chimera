from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anti_chimera.config import load_config
from anti_chimera.data.manifest import ManifestVideoDataset
from anti_chimera.data.scene_hint_modes import build_scene_hint_builder
from anti_chimera.inference import sample_video
from anti_chimera.inference_with_planner import sample_video_with_planner
from anti_chimera.metrics import compute_chimera_metrics
from anti_chimera.models.model import AntiChimeraVideoDiffusion
from anti_chimera.trainer import collate_fn, build_batch_cond
from anti_chimera.utils import default_device


def _avg_dict(items: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        for key, value in item.items():
            out[key] = out.get(key, 0.0) + float(value)
    if not items:
        return out
    return {key: value / len(items) for key, value in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--limit', default=16, type=int)
    parser.add_argument('--planner-checkpoint', default=None, type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.planner_checkpoint is not None:
        config.setdefault('planner', {})['checkpoint'] = args.planner_checkpoint

    data_cfg = config['data']
    model_cfg = config['model']
    planner_checkpoint = dict(config.get('planner', {})).get('checkpoint')
    device = default_device(config.get('training', {}).get('device', 'cuda'))

    ds = ManifestVideoDataset(
        manifest_path=data_cfg.get('val_manifest_path') or data_cfg['manifest_path'],
        root_dir=data_cfg.get('root_dir', '.'),
        num_frames=int(data_cfg['num_frames']),
        image_size=int(data_cfg['image_size']),
        max_objects=int(data_cfg['max_objects']),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    builder = build_scene_hint_builder(data_cfg)
    model = AntiChimeraVideoDiffusion(cond_channels=builder.num_channels(), **model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    oracle_metrics_all: list[dict[str, float]] = []
    prompt_only_metrics_all: list[dict[str, float]] = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.limit:
                break
            prompt = batch['caption'][0]
            oracle_cond = build_batch_cond(batch, builder).float().to(device)[:1]
            oracle_video = sample_video(model, prompt, config, device, cond=oracle_cond)
            if planner_checkpoint:
                prompt_only_video = sample_video_with_planner(model, prompt, config, device, planner_checkpoint=planner_checkpoint)
            else:
                prompt_only_video = sample_video(model, prompt, config, device, cond=None)
            oracle_metrics_all.append(
                compute_chimera_metrics(
                    target_video=batch['video'][0].float(),
                    generated_video=oracle_video.float(),
                    tracks=batch['tracks'][0].float(),
                    visibility=batch['visibility'][0].float(),
                )
            )
            prompt_only_metrics_all.append(
                compute_chimera_metrics(
                    target_video=batch['video'][0].float(),
                    generated_video=prompt_only_video.float(),
                    tracks=batch['tracks'][0].float(),
                    visibility=batch['visibility'][0].float(),
                )
            )

    payload = {
        'num_samples': min(args.limit, len(ds)),
        'planner_checkpoint': planner_checkpoint,
        'oracle': _avg_dict(oracle_metrics_all),
        'prompt_only': _avg_dict(prompt_only_metrics_all),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
