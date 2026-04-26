from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from anti_chimera.data.manifest import ManifestVideoDataset
from anti_chimera.data.scene_hint import SceneHintBuilder
from anti_chimera.inference import sample_video
from anti_chimera.metrics import compute_chimera_metrics
from anti_chimera.models.model import AntiChimeraVideoDiffusion
from anti_chimera.utils import default_device, ensure_dir, normalize_video, save_gif, save_video_png


def _collate_fn(batch):
    out = {
        'video': torch.stack([item['video'] for item in batch]),
        'caption': [item['caption'] for item in batch],
        'tracks': torch.stack([item['tracks'] for item in batch]),
        'depth': torch.stack([item['depth'] for item in batch]),
        'visibility': torch.stack([item['visibility'] for item in batch]),
    }
    for key in ('instance_map', 'masks', 'flow', 'occlusion'):
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch])
    return out


def _batch_to_condition(batch: Dict, builder: SceneHintBuilder, device: torch.device) -> torch.Tensor:
    conds = []
    for i in range(batch['video'].shape[0]):
        sample = {
            'video': batch['video'][i],
            'caption': batch['caption'][i],
            'tracks': batch['tracks'][i],
            'depth': batch['depth'][i],
            'visibility': batch['visibility'][i],
        }
        for key in ('instance_map', 'masks', 'flow', 'occlusion'):
            if key in batch:
                sample[key] = batch[key][i]
        conds.append(builder.build(sample))
    return torch.stack(conds, dim=0).to(device)


def train(config: Dict, resume_checkpoint: str | None = None) -> None:
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    device = default_device(train_cfg.get('device', 'cuda'))

    out_dir = ensure_dir(config['output_dir'])
    ckpt_dir = ensure_dir(out_dir / 'checkpoints')
    sample_dir = ensure_dir(out_dir / 'samples')
    metrics_path = out_dir / 'metrics.jsonl'

    if not Path(data_cfg['manifest_path']).exists():
        raise FileNotFoundError('trainer_cogvideox_v2 expects a manifest dataset. Build one with scripts/build_manifest.py.')

    train_ds = ManifestVideoDataset(
        manifest_path=data_cfg['manifest_path'],
        root_dir=data_cfg.get('root_dir', '.'),
        num_frames=int(data_cfg['num_frames']),
        image_size=int(data_cfg['image_size']),
        max_objects=int(data_cfg['max_objects']),
    )
    val_ds = ManifestVideoDataset(
        manifest_path=data_cfg.get('val_manifest_path') or data_cfg['manifest_path'],
        root_dir=data_cfg.get('root_dir', '.'),
        num_frames=int(data_cfg['num_frames']),
        image_size=int(data_cfg['image_size']),
        max_objects=int(data_cfg['max_objects']),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get('batch_size', 1)),
        shuffle=True,
        num_workers=int(train_cfg.get('num_workers', 0)),
        collate_fn=_collate_fn,
        pin_memory=device.type == 'cuda',
        persistent_workers=int(train_cfg.get('num_workers', 0)) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(train_cfg.get('num_workers', 0)),
        collate_fn=_collate_fn,
        pin_memory=device.type == 'cuda',
        persistent_workers=False,
    )

    builder = SceneHintBuilder(
        max_objects=int(data_cfg['max_objects']),
        depth_bins=int(data_cfg['depth_bins']),
        image_size=int(data_cfg['image_size']),
    )
    model = AntiChimeraVideoDiffusion(cond_channels=builder.num_channels(), **model_cfg).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(train_cfg.get('lr', 1e-4)), weight_decay=float(train_cfg.get('weight_decay', 0.0)))
    scaler = GradScaler(enabled=bool(train_cfg.get('amp', False)) and device.type == 'cuda')

    start_epoch = 1
    global_step = 0
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        global_step = int(ckpt.get('step', 0))

    with open(out_dir / 'run_log.md', 'w', encoding='utf-8') as f:
        f.write('# CogVideoX conditioner-only training\n\n')
        f.write(f'- trainable_parameters: `{sum(int(p.numel()) for p in trainable_params)}`\n')
        f.write(f'- cond_channels: `{builder.num_channels()}`\n')
        f.write(f'- device: `{device}`\n')

    for epoch in range(start_epoch, int(train_cfg.get('epochs', 1)) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'CogVideoX Epoch {epoch}', leave=False)
        losses = []
        for batch in pbar:
            videos = normalize_video(batch['video'].float().to(device))
            prompts = batch['caption']
            cond = _batch_to_condition(batch, builder, device).float()

            with torch.no_grad():
                latents = model.encode_video(videos)
            timesteps = torch.randint(0, model.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(latents)
            noisy_latents = model.add_noise(latents, noise, timesteps)
            target = model.prediction_target(latents, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(train_cfg.get('amp', False)) and device.type == 'cuda'):
                pred = model(noisy_latents, timesteps, prompts, cond)
                loss = F.mse_loss(pred, target)
            scaler.scale(loss).backward()
            if train_cfg.get('grad_clip') is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, float(train_cfg['grad_clip']))
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.4f}', step=global_step)

        avg_loss = float(sum(losses) / max(len(losses), 1))
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
            'step': global_step,
        }
        torch.save(ckpt, ckpt_dir / 'last.pt')
        torch.save(ckpt, ckpt_dir / f'epoch_{epoch:03d}.pt')

        chimera_metrics = None
        if epoch % int(train_cfg.get('sample_every', 1)) == 0:
            model.eval()
            batch = next(iter(val_loader))
            target = normalize_video(batch['video'][0])
            save_gif(target, sample_dir / f'epoch_{epoch:03d}_target.gif')
            save_video_png(target, sample_dir / f'epoch_{epoch:03d}_target.png')
            cond = _batch_to_condition(batch, builder, device).float()[:1]
            generated = sample_video(model, batch['caption'][0], config, device, cond=cond)
            save_gif(generated * 2 - 1, sample_dir / f'epoch_{epoch:03d}_sample.gif')
            save_video_png(generated * 2 - 1, sample_dir / f'epoch_{epoch:03d}_sample.png')
            chimera_metrics = compute_chimera_metrics(
                target_video=batch['video'][0].float(),
                generated_video=generated.float(),
                tracks=batch['tracks'][0].float(),
                visibility=batch['visibility'][0].float(),
            )

        payload = {'epoch': epoch, 'step': global_step, 'train_loss': avg_loss}
        if chimera_metrics is not None:
            payload.update({f'chimera/{k}': v for k, v in chimera_metrics.items()})
        with open(metrics_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
