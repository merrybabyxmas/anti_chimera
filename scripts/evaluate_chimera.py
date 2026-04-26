from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anti_chimera.config import load_config
from anti_chimera.data.manifest import ManifestVideoDataset
from anti_chimera.data.scene_hint import SceneHintBuilder
from anti_chimera.inference import sample_video
from anti_chimera.metrics import compute_chimera_metrics
from anti_chimera.models.model import AntiChimeraVideoDiffusion
from anti_chimera.trainer import collate_fn, build_batch_cond
from anti_chimera.utils import default_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--limit', type=int, default=16)
    args = parser.parse_args()

    config = load_config(args.config)
    device = default_device(config.get('training', {}).get('device', 'cuda'))
    data_cfg = config['data']
    model_cfg = config['model']

    ds = ManifestVideoDataset(
        manifest_path=data_cfg['val_manifest_path'] or data_cfg['manifest_path'],
        root_dir=data_cfg.get('root_dir', '.'),
        num_frames=int(data_cfg['num_frames']),
        image_size=int(data_cfg['image_size']),
        max_objects=int(data_cfg['max_objects']),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    builder = SceneHintBuilder(
        max_objects=int(data_cfg['max_objects']),
        depth_bins=int(data_cfg['depth_bins']),
        image_size=int(data_cfg['image_size']),
    )
    model = AntiChimeraVideoDiffusion(cond_channels=builder.num_channels(), **model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            cond = build_batch_cond(batch, builder).float().to(device)[:1]
            generated = sample_video(model, batch['caption'][0], config, device, cond=cond)
            metrics = compute_chimera_metrics(
                target_video=batch['video'][0].float(),
                generated_video=generated.float(),
                tracks=batch['tracks'][0].float(),
                visibility=batch['visibility'][0].float(),
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            count += 1
            if count >= args.limit:
                break

    for key in sorted(totals):
        print(f'{key}: {totals[key] / max(count, 1):.6f}')


if __name__ == '__main__':
    main()
