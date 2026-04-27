from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from anti_chimera.data.manifest import ManifestVideoDataset
from anti_chimera.data.synthetic_collision import SyntheticCollisionDataset
from anti_chimera.planner_learned import LearnedPromptPlanner, planner_losses
from anti_chimera.utils import default_device, ensure_dir, set_seed


def _collate(batch):
    out = {
        'caption': [item['caption'] for item in batch],
        'tracks': torch.stack([item['tracks'] for item in batch]),
        'depth': torch.stack([item['depth'] for item in batch]),
        'visibility': torch.stack([item['visibility'] for item in batch]),
    }
    for key in ('masks', 'occlusion'):
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch])
    return out


def _build_datasets(config: Dict):
    data_cfg = config['data']
    data_type = str(data_cfg.get('type', 'synthetic'))
    if data_type == 'manifest' and Path(data_cfg['manifest_path']).exists():
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
        return train_ds, val_ds
    difficulty = str(data_cfg.get('synthetic_difficulty', 'mixed'))
    train_ds = SyntheticCollisionDataset(
        size=int(data_cfg.get('train_size', 12000)),
        num_frames=int(data_cfg['num_frames']),
        image_size=int(data_cfg['image_size']),
        max_objects=int(data_cfg['max_objects']),
        seed=int(config.get('seed', 42)),
        difficulty=difficulty,
    )
    val_ds = SyntheticCollisionDataset(
        size=int(data_cfg.get('val_size', 256)),
        num_frames=int(data_cfg['num_frames']),
        image_size=int(data_cfg['image_size']),
        max_objects=int(data_cfg['max_objects']),
        seed=int(config.get('seed', 42)) + 1,
        difficulty=difficulty,
    )
    return train_ds, val_ds


def _planner_metric(pred_tracks: torch.Tensor, gt_tracks: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred_tracks - gt_tracks)).item()


def train_planner(config: Dict, resume_checkpoint: str | None = None) -> None:
    seed = int(config.get('seed', 42))
    set_seed(seed)
    data_cfg = config['data']
    planner_cfg = dict(config.get('planner', {}))
    train_cfg = config['training']
    device = default_device(train_cfg.get('device', 'cuda'))

    out_dir = ensure_dir(config['output_dir'])
    ckpt_dir = ensure_dir(out_dir / 'checkpoints')
    metrics_path = out_dir / 'planner_metrics.jsonl'

    train_ds, val_ds = _build_datasets(config)
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg.get('batch_size', 16)), shuffle=True, num_workers=int(train_cfg.get('num_workers', 0)), collate_fn=_collate, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg.get('eval_batch_size', train_cfg.get('batch_size', 16))), shuffle=False, num_workers=int(train_cfg.get('num_workers', 0)), collate_fn=_collate, pin_memory=device.type == 'cuda')

    planner = LearnedPromptPlanner(
        max_objects=int(data_cfg['max_objects']),
        num_frames=int(data_cfg['num_frames']),
        image_size=int(data_cfg['image_size']),
        prompt_vocab_size=int(planner_cfg.get('prompt_vocab_size', 4096)),
        hidden_size=int(planner_cfg.get('hidden_size', 256)),
        max_prompt_tokens=int(planner_cfg.get('max_prompt_tokens', 48)),
        num_layers=int(planner_cfg.get('num_layers', 4)),
        num_heads=int(planner_cfg.get('num_heads', 8)),
        dropout=float(planner_cfg.get('dropout', 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(planner.parameters(), lr=float(train_cfg.get('lr', 1e-4)), weight_decay=float(train_cfg.get('weight_decay', 1e-4)))
    scaler = GradScaler(enabled=bool(train_cfg.get('amp', False)) and device.type == 'cuda')

    start_epoch = 1
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        planner.load_state_dict(ckpt['planner'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1

    for epoch in range(start_epoch, int(train_cfg.get('epochs', 20)) + 1):
        planner.train()
        losses = []
        pbar = tqdm(train_loader, desc=f'Planner Epoch {epoch}', leave=False)
        for batch in pbar:
            gt_tracks = batch['tracks'].float().to(device)
            gt_depth = batch['depth'].float().to(device)
            gt_visibility = batch['visibility'].float().to(device)
            gt_masks = batch.get('masks', torch.zeros(gt_tracks.shape[0], gt_tracks.shape[1], gt_tracks.shape[2], int(data_cfg['image_size']), int(data_cfg['image_size']))).float().to(device)
            gt_occlusion = batch.get('occlusion', torch.zeros(gt_tracks.shape[0], gt_tracks.shape[1], int(data_cfg['image_size']), int(data_cfg['image_size']))).float().to(device)
            gt_count = gt_visibility.amax(dim=1).sum(dim=1).clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(train_cfg.get('amp', False)) and device.type == 'cuda'):
                pred = planner(batch['caption'], device=device)
                loss_dict = planner_losses(
                    pred,
                    gt_tracks=gt_tracks,
                    gt_depth=gt_depth,
                    gt_visibility=gt_visibility,
                    gt_masks=gt_masks,
                    gt_occlusion=gt_occlusion,
                    gt_count=gt_count,
                    mask_weight=float(train_cfg.get('mask_weight', 1.0)),
                    occ_weight=float(train_cfg.get('occ_weight', 1.0)),
                    track_weight=float(train_cfg.get('track_weight', 5.0)),
                    depth_weight=float(train_cfg.get('depth_weight', 2.0)),
                    visibility_weight=float(train_cfg.get('visibility_weight', 1.0)),
                    count_weight=float(train_cfg.get('count_weight', 0.5)),
                )
            scaler.scale(loss_dict['total']).backward()
            if train_cfg.get('grad_clip') is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(planner.parameters(), float(train_cfg['grad_clip']))
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss_dict['total'].item())
            pbar.set_postfix(loss=f"{loss_dict['total'].item():.4f}")

        planner.eval()
        val_track = []
        val_total = []
        with torch.no_grad():
            for batch in val_loader:
                gt_tracks = batch['tracks'].float().to(device)
                gt_depth = batch['depth'].float().to(device)
                gt_visibility = batch['visibility'].float().to(device)
                gt_masks = batch.get('masks', torch.zeros(gt_tracks.shape[0], gt_tracks.shape[1], gt_tracks.shape[2], int(data_cfg['image_size']), int(data_cfg['image_size']))).float().to(device)
                gt_occlusion = batch.get('occlusion', torch.zeros(gt_tracks.shape[0], gt_tracks.shape[1], int(data_cfg['image_size']), int(data_cfg['image_size']))).float().to(device)
                gt_count = gt_visibility.amax(dim=1).sum(dim=1).clamp_min(1)
                pred = planner(batch['caption'], device=device)
                loss_dict = planner_losses(
                    pred,
                    gt_tracks=gt_tracks,
                    gt_depth=gt_depth,
                    gt_visibility=gt_visibility,
                    gt_masks=gt_masks,
                    gt_occlusion=gt_occlusion,
                    gt_count=gt_count,
                    mask_weight=float(train_cfg.get('mask_weight', 1.0)),
                    occ_weight=float(train_cfg.get('occ_weight', 1.0)),
                    track_weight=float(train_cfg.get('track_weight', 5.0)),
                    depth_weight=float(train_cfg.get('depth_weight', 2.0)),
                    visibility_weight=float(train_cfg.get('visibility_weight', 1.0)),
                    count_weight=float(train_cfg.get('count_weight', 0.5)),
                )
                val_total.append(loss_dict['total'].item())
                val_track.append(_planner_metric(pred.tracks, gt_tracks))

        payload = {'epoch': epoch, 'train_loss': float(sum(losses) / max(len(losses), 1)), 'val_loss': float(sum(val_total) / max(len(val_total), 1)), 'val_track_l1': float(sum(val_track) / max(len(val_track), 1))}
        with open(metrics_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        torch.save({'planner': planner.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'config': config}, ckpt_dir / 'last.pt')
        torch.save({'planner': planner.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'config': config}, ckpt_dir / f'epoch_{epoch:03d}.pt')
