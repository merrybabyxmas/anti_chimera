from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from anti_chimera.data.manifest import ManifestVideoDataset
from anti_chimera.data.scene_hint import SceneHintBuilder
from anti_chimera.data.synthetic_collision import SyntheticCollisionDataset
from anti_chimera.models.model import AntiChimeraVideoDiffusion
from anti_chimera.utils import cosine_beta_schedule, default_device, ensure_dir, normalize_video, save_gif


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor


def make_schedule(num_steps: int, device: torch.device) -> DiffusionSchedule:
    betas = cosine_beta_schedule(num_steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
    )


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, schedule: DiffusionSchedule) -> torch.Tensor:
    a = schedule.sqrt_alphas_cumprod[t][:, None, None, None, None]
    b = schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]
    return a * x0 + b * noise


def collate_fn(batch):
    videos = torch.stack([item['video'] for item in batch])
    captions = [item['caption'] for item in batch]
    tracks = torch.stack([item['tracks'] for item in batch])
    depth = torch.stack([item['depth'] for item in batch])
    visibility = torch.stack([item['visibility'] for item in batch])
    return {'video': videos, 'caption': captions, 'tracks': tracks, 'depth': depth, 'visibility': visibility}


def build_batch_cond(batch: Dict, builder: SceneHintBuilder) -> torch.Tensor:
    conds = []
    B = batch['video'].shape[0]
    for i in range(B):
        sample = {
            'video': batch['video'][i],
            'caption': batch['caption'][i],
            'tracks': batch['tracks'][i],
            'depth': batch['depth'][i],
            'visibility': batch['visibility'][i],
        }
        conds.append(builder.build(sample))
    return torch.stack(conds, dim=0)


def build_datasets(config: Dict):
    data_cfg = config['data']
    data_type = str(data_cfg.get('type', 'manifest'))
    if data_type == 'manifest' and Path(data_cfg['manifest_path']).exists():
        train_ds = ManifestVideoDataset(
            manifest_path=data_cfg['manifest_path'],
            root_dir=data_cfg.get('root_dir', '.'),
            num_frames=data_cfg['num_frames'],
            image_size=data_cfg['image_size'],
            max_objects=data_cfg['max_objects'],
        )
        val_manifest = data_cfg.get('val_manifest_path') or data_cfg['manifest_path']
        val_ds = ManifestVideoDataset(
            manifest_path=val_manifest,
            root_dir=data_cfg.get('root_dir', '.'),
            num_frames=data_cfg['num_frames'],
            image_size=data_cfg['image_size'],
            max_objects=data_cfg['max_objects'],
        )
        return train_ds, val_ds

    if not data_cfg.get('synthetic_fallback', True):
        raise FileNotFoundError(
            f"Manifest dataset not found at {data_cfg['manifest_path']}. Set data.synthetic_fallback=true or provide a manifest."
        )

    train_ds = SyntheticCollisionDataset(
        size=data_cfg['train_size'],
        num_frames=data_cfg['num_frames'],
        image_size=data_cfg['image_size'],
        max_objects=data_cfg['max_objects'],
        seed=config['seed'],
    )
    val_ds = SyntheticCollisionDataset(
        size=data_cfg['val_size'],
        num_frames=data_cfg['num_frames'],
        image_size=data_cfg['image_size'],
        max_objects=data_cfg['max_objects'],
        seed=config['seed'] + 1,
    )
    return train_ds, val_ds


def train(config: Dict) -> None:
    device = default_device(config['training']['device'])
    out_dir = ensure_dir(config['output_dir'])
    ckpt_dir = ensure_dir(out_dir / 'checkpoints')
    sample_dir = ensure_dir(out_dir / 'train_samples')

    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']

    train_ds, val_ds = build_datasets(config)
    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'], collate_fn=collate_fn)

    builder = SceneHintBuilder(
        max_objects=data_cfg['max_objects'],
        depth_bins=data_cfg['depth_bins'],
        image_size=data_cfg['image_size'],
    )
    cond_channels = data_cfg['max_objects'] + data_cfg['depth_bins'] + data_cfg['max_objects']
    model = AntiChimeraVideoDiffusion(
        cond_channels=cond_channels,
        **model_cfg,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    scaler = GradScaler(enabled=bool(train_cfg['amp']) and device.type == 'cuda')
    schedule = make_schedule(config['sampling']['num_steps'], device)

    for epoch in range(1, train_cfg['epochs'] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            videos = normalize_video(batch['video'].float().to(device))
            prompts = batch['caption']
            cond = build_batch_cond(batch, builder).float().to(device)
            B = videos.shape[0]
            t = torch.randint(0, config['sampling']['num_steps'], (B,), device=device)
            noise = torch.randn_like(videos)
            x_t = q_sample(videos, t, noise, schedule)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(train_cfg['amp']) and device.type == 'cuda'):
                pred = model(x_t, t, prompts, cond)
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()
            if train_cfg['grad_clip'] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        torch.save(ckpt, ckpt_dir / 'last.pt')
        torch.save(ckpt, ckpt_dir / f'epoch_{epoch:03d}.pt')

        model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            preview = normalize_video(batch['video'][0])
            save_gif(preview, sample_dir / f'epoch_{epoch:03d}_target.gif')
