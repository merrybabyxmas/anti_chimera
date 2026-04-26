from __future__ import annotations

import math
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from anti_chimera.data.manifest import ManifestVideoDataset
from anti_chimera.data.scene_hint import SceneHintBuilder
from anti_chimera.data.synthetic_collision import SyntheticCollisionDataset
from anti_chimera.inference import sample_video
from anti_chimera.models.model import AntiChimeraVideoDiffusion
from anti_chimera.utils import default_device, ensure_dir, normalize_video, save_gif, save_video_png


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


def _sample_timesteps(batch_size: int, num_train_timesteps: int, device: torch.device, power: float) -> torch.Tensor:
    if power <= 1.0:
        return torch.randint(0, num_train_timesteps, (batch_size,), device=device, dtype=torch.long)
    u = torch.rand(batch_size, device=device)
    timesteps = (u.pow(power) * (num_train_timesteps - 1)).long()
    return timesteps.clamp_(0, num_train_timesteps - 1)


def _snr_weights(
    model: AntiChimeraVideoDiffusion,
    timesteps: torch.Tensor,
    strength: float,
    snr_cap: float,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    if strength <= 0:
        return torch.ones_like(timesteps, dtype=torch.float32)
    if not hasattr(model.scheduler, 'alphas_cumprod'):
        return torch.ones_like(timesteps, dtype=torch.float32)
    alphas_cumprod = model.scheduler.alphas_cumprod.to(device=timesteps.device)
    alpha = alphas_cumprod.gather(0, timesteps.long())
    snr = alpha / (1.0 - alpha).clamp_min(1e-8)
    snr = snr.clamp(max=snr_cap)
    normalized = torch.log1p(snr) / math.log1p(snr_cap)
    weights = 1.0 + strength * (normalized - 0.5)
    return weights.clamp(min=min_weight, max=max_weight).to(dtype=torch.float32)


def _weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    model: AntiChimeraVideoDiffusion,
    timesteps: torch.Tensor,
    train_cfg: Dict,
) -> torch.Tensor:
    per_sample = (pred - target).pow(2).flatten(1).mean(dim=1)
    weights = _snr_weights(
        model,
        timesteps,
        strength=float(train_cfg.get('snr_weight_strength', 0.0)),
        snr_cap=float(train_cfg.get('snr_weight_cap', 20.0)),
        min_weight=float(train_cfg.get('snr_weight_min', 0.75)),
        max_weight=float(train_cfg.get('snr_weight_max', 1.25)),
    )
    return (per_sample * weights).mean()


def _clone_state_dict(state: Dict[str, torch.Tensor], keys: set[str] | None = None) -> Dict[str, torch.Tensor]:
    cloned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if keys is not None and key not in keys:
            continue
        cloned[key] = value.detach().clone() if torch.is_tensor(value) else value
    return cloned


def _trainable_state_keys(model: AntiChimeraVideoDiffusion) -> set[str]:
    return {name for name, param in model.named_parameters() if param.requires_grad}


def _init_ema_state(model: AntiChimeraVideoDiffusion) -> Dict[str, torch.Tensor]:
    return _clone_state_dict(model.state_dict(), keys=_trainable_state_keys(model))


def _update_ema_state(ema_state: Dict[str, torch.Tensor], model: AntiChimeraVideoDiffusion, decay: float) -> None:
    if not (0.0 < decay < 1.0):
        return
    model_state = model.state_dict()
    for key, value in ema_state.items():
        source = model_state.get(key)
        if source is None or not torch.is_tensor(source) or not source.is_floating_point():
            continue
        value.mul_(decay).add_(source.detach(), alpha=1.0 - decay)


def _load_state_dict(model: AntiChimeraVideoDiffusion, state: Dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    for key, value in state.items():
        if key in current:
            current[key] = value.detach().clone() if torch.is_tensor(value) else value
    model.load_state_dict(current, strict=False)


def _maybe_drop_condition(cond: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0:
        return cond
    mask = torch.rand(cond.shape[0], device=cond.device) < drop_prob
    if mask.any():
        cond = cond.clone()
        cond[mask] = 0
    return cond


def _maybe_drop_prompts(prompts: Iterable[str], drop_prob: float) -> list[str]:
    if drop_prob <= 0:
        return list(prompts)
    out = []
    for prompt in prompts:
        out.append('' if random.random() < drop_prob else prompt)
    return out


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

    if data_type not in {'synthetic', 'manifest'} and not data_cfg.get('synthetic_fallback', True):
        raise FileNotFoundError(
            f"Manifest dataset not found at {data_cfg['manifest_path']}. Set data.synthetic_fallback=true or provide a manifest."
        )

    train_ds = SyntheticCollisionDataset(
        size=int(data_cfg['train_size']),
        num_frames=int(data_cfg['num_frames']),
        image_size=int(data_cfg['image_size']),
        max_objects=int(data_cfg['max_objects']),
        seed=int(config['seed']),
    )
    val_ds = SyntheticCollisionDataset(
        size=int(data_cfg['val_size']),
        num_frames=int(data_cfg['num_frames']),
        image_size=int(data_cfg['image_size']),
        max_objects=int(data_cfg['max_objects']),
        seed=int(config['seed']) + 1,
    )
    return train_ds, val_ds


def _val_loss(
    model: AntiChimeraVideoDiffusion,
    val_loader: DataLoader,
    builder: SceneHintBuilder,
    device: torch.device,
    train_cfg: Dict,
    limit_batches: int | None = None,
) -> Tuple[float, float | None, float | None]:
    model.eval()
    losses = []
    low_noise_losses = []
    high_noise_losses = []
    low_t_threshold = max(1, int(model.num_train_timesteps * float(train_cfg.get('low_noise_eval_fraction', 0.25))))
    high_t_threshold = min(model.num_train_timesteps - 1, int(model.num_train_timesteps * float(train_cfg.get('high_noise_eval_fraction', 0.75))))
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            videos = normalize_video(batch['video'].float().to(device))
            prompts = batch['caption']
            cond = build_batch_cond(batch, builder).float().to(device)
            cond = _maybe_drop_condition(cond, float(train_cfg.get('condition_dropout_prob', 0.0)) * 0.25)
            prompts = _maybe_drop_prompts(prompts, float(train_cfg.get('prompt_dropout_prob', 0.0)) * 0.25)
            latents = model.encode_video(videos)
            B = latents.shape[0]
            t = _sample_timesteps(B, model.num_train_timesteps, device, power=1.0)
            noise = torch.randn_like(latents)
            noisy_latents = model.add_noise(latents, noise, t)
            target = model.prediction_target(latents, noise, t)
            pred = model(noisy_latents, t, prompts, cond)
            loss = _weighted_mse_loss(pred, target, model, t, train_cfg)
            losses.append(loss.item())

            per_sample = (pred - target).pow(2).flatten(1).mean(dim=1)
            weights = _snr_weights(
                model,
                t,
                strength=float(train_cfg.get('snr_weight_strength', 0.0)),
                snr_cap=float(train_cfg.get('snr_weight_cap', 20.0)),
                min_weight=float(train_cfg.get('snr_weight_min', 0.75)),
                max_weight=float(train_cfg.get('snr_weight_max', 1.25)),
            )
            weighted = per_sample * weights
            low_mask = t <= low_t_threshold
            high_mask = t >= high_t_threshold
            if low_mask.any():
                low_noise_losses.append(weighted[low_mask].mean().item())
            if high_mask.any():
                high_noise_losses.append(weighted[high_mask].mean().item())
    val_loss = float(sum(losses) / max(len(losses), 1))
    low_loss = float(sum(low_noise_losses) / max(len(low_noise_losses), 1)) if low_noise_losses else None
    high_loss = float(sum(high_noise_losses) / max(len(high_noise_losses), 1)) if high_noise_losses else None
    return val_loss, low_loss, high_loss


def train(config: Dict, resume_checkpoint: str | None = None) -> None:
    device = default_device(config['training']['device'])
    out_dir = ensure_dir(config['output_dir'])
    ckpt_dir = ensure_dir(out_dir / 'checkpoints')
    sample_dir = ensure_dir(out_dir / 'train_samples')
    log_path = out_dir / 'metrics.jsonl'
    run_log = out_dir / 'run_log.md'

    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']

    train_ds, val_ds = build_datasets(config)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg['batch_size']),
        shuffle=True,
        num_workers=int(train_cfg['num_workers']),
        collate_fn=collate_fn,
        pin_memory=device.type == 'cuda',
        persistent_workers=int(train_cfg['num_workers']) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get('eval_batch_size', train_cfg['batch_size'])),
        shuffle=False,
        num_workers=int(train_cfg['num_workers']),
        collate_fn=collate_fn,
        pin_memory=device.type == 'cuda',
        persistent_workers=int(train_cfg['num_workers']) > 0,
    )

    if str(model_cfg.get('backend', 'lite3d')) == 'cogvideox':
        torch.backends.cudnn.enabled = False

    builder = SceneHintBuilder(
        max_objects=int(data_cfg['max_objects']),
        depth_bins=int(data_cfg['depth_bins']),
        image_size=int(data_cfg['image_size']),
    )
    cond_channels = int(data_cfg['max_objects']) + int(data_cfg['depth_bins']) + int(data_cfg['max_objects'])
    model = AntiChimeraVideoDiffusion(cond_channels=cond_channels, **model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg['lr']),
        weight_decay=float(train_cfg['weight_decay']),
    )
    scaler = GradScaler(enabled=bool(train_cfg.get('amp', False)) and device.type == 'cuda')

    start_epoch = 1
    max_steps = int(train_cfg.get('max_steps', 0) or 0)
    grad_clip = train_cfg.get('grad_clip', None)
    condition_dropout_prob = float(train_cfg.get('condition_dropout_prob', 0.0))
    prompt_dropout_prob = float(train_cfg.get('prompt_dropout_prob', 0.0))
    val_every = max(1, int(train_cfg.get('val_every', 1)))
    sample_every = max(1, int(train_cfg.get('sample_every', 1)))
    val_batches = train_cfg.get('val_batches', None)
    val_batches = None if val_batches is None else int(val_batches)
    timestep_sampling_power = float(train_cfg.get('timestep_sampling_power', 1.5))
    snr_weight_strength = float(train_cfg.get('snr_weight_strength', 0.0))
    snr_weight_cap = float(train_cfg.get('snr_weight_cap', 20.0))
    snr_weight_min = float(train_cfg.get('snr_weight_min', 0.75))
    snr_weight_max = float(train_cfg.get('snr_weight_max', 1.25))
    ema_decay = float(train_cfg.get('ema_decay', 0.999))
    ema_start_step = int(train_cfg.get('ema_start_step', 0))
    ema_enabled = 0.0 < ema_decay < 1.0

    global_step = 0
    ema_state = _init_ema_state(model) if ema_enabled else None
    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        state = dict(ckpt['model'])
        for key in list(state.keys()):
            if key.endswith('scheduler.timesteps'):
                state.pop(key)
        model.load_state_dict(state, strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        global_step = int(ckpt.get('step', 0))
        if ema_enabled:
            ema_state = ckpt.get('ema_model') or _init_ema_state(model)

    with open(run_log, 'w', encoding='utf-8') as f:
        f.write('# anti_chimera run log\n\n')
        f.write(f'- device: `{device}`\n')
        f.write(f'- backend: `{model_cfg.get("backend", "lite3d")}`\n')
        f.write(f'- output_dir: `{out_dir}`\n')
        f.write(f'- train_size: `{len(train_ds)}`\n')
        f.write(f'- val_size: `{len(val_ds)}`\n')
        f.write(f'- max_steps: `{max_steps or "unbounded"}`\n')
        f.write(f'- condition_dropout_prob: `{condition_dropout_prob}`\n')
        f.write(f'- prompt_dropout_prob: `{prompt_dropout_prob}`\n\n')
        f.write(f'- timestep_sampling_power: `{timestep_sampling_power}`\n')
        f.write(f'- snr_weight_strength: `{snr_weight_strength}`\n')
        f.write(f'- snr_weight_cap: `{snr_weight_cap}`\n')
        f.write(f'- snr_weight_min: `{snr_weight_min}`\n')
        f.write(f'- snr_weight_max: `{snr_weight_max}`\n')
        f.write(f'- ema_decay: `{ema_decay}`\n')
        f.write(f'- ema_start_step: `{ema_start_step}`\n\n')
        if resume_checkpoint is not None:
            f.write(f'- resumed_from: `{resume_checkpoint}`\n\n')

    for epoch in range(start_epoch, int(train_cfg['epochs']) + 1):
        model.train()
        running_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        for batch in pbar:
            videos = normalize_video(batch['video'].float().to(device))
            prompts = _maybe_drop_prompts(batch['caption'], prompt_dropout_prob)
            cond = build_batch_cond(batch, builder).float().to(device)
            cond = _maybe_drop_condition(cond, condition_dropout_prob)

            with torch.no_grad():
                latents = model.encode_video(videos)
            B = latents.shape[0]
            t = _sample_timesteps(B, model.num_train_timesteps, device, power=timestep_sampling_power)
            noise = torch.randn_like(latents)
            noisy_latents = model.add_noise(latents, noise, t)
            target = model.prediction_target(latents, noise, t)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(train_cfg.get('amp', False)) and device.type == 'cuda'):
                pred = model(noisy_latents, t, prompts, cond)
                loss = _weighted_mse_loss(pred, target, model, t, train_cfg)

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if ema_state is not None and global_step >= ema_start_step:
                _update_ema_state(ema_state, model, ema_decay)
            running_losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.4f}', step=global_step)

            if max_steps and global_step >= max_steps:
                break

        avg_train_loss = float(sum(running_losses) / max(len(running_losses), 1))
        if epoch % val_every == 0:
            val_loss, val_low_t_loss, val_high_t_loss = _val_loss(model, val_loader, builder, device, train_cfg, limit_batches=val_batches)
        else:
            val_loss, val_low_t_loss, val_high_t_loss = None, None, None

        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
            'step': global_step,
            'backend': model_cfg.get('backend', 'lite3d'),
        }
        if ema_state is not None:
            ckpt['ema_model'] = ema_state
        torch.save(ckpt, ckpt_dir / 'last.pt')
        torch.save(ckpt, ckpt_dir / f'epoch_{epoch:03d}.pt')

        if epoch % sample_every == 0:
            model.eval()
            with torch.no_grad():
                batch = next(iter(val_loader))
                preview = normalize_video(batch['video'][0])
                target_gif = sample_dir / f'epoch_{epoch:03d}_target.gif'
                target_png = sample_dir / f'epoch_{epoch:03d}_target.png'
                save_gif(preview, target_gif)
                save_video_png(preview, target_png)
                prompt = batch['caption'][0]
                preview_cond = build_batch_cond(batch, builder).float().to(device)[:1]
                backup_state = _clone_state_dict(model.state_dict(), keys=set(ema_state.keys())) if ema_state is not None else None
                if ema_state is not None:
                    _load_state_dict(model, ema_state)
                generated = sample_video(model, prompt, config, device, cond=preview_cond)
                if backup_state is not None:
                    _load_state_dict(model, backup_state)
                sample_gif = sample_dir / f'epoch_{epoch:03d}_sample.gif'
                sample_png = sample_dir / f'epoch_{epoch:03d}_sample.png'
                save_gif(generated * 2 - 1, sample_gif)
                save_video_png(generated * 2 - 1, sample_png)

        metrics = {
            'epoch': epoch,
            'step': global_step,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_loss_low_t': val_low_t_loss,
            'val_loss_high_t': val_high_t_loss,
        }
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + '\n')

        with open(run_log, 'a', encoding='utf-8') as f:
            f.write(f'## Epoch {epoch}\n\n')
            f.write(f'- train_loss: `{avg_train_loss:.6f}`\n')
            if val_loss is not None:
                f.write(f'- val_loss: `{val_loss:.6f}`\n')
                if val_low_t_loss is not None:
                    f.write(f'- val_loss_low_t: `{val_low_t_loss:.6f}`\n')
                if val_high_t_loss is not None:
                    f.write(f'- val_loss_high_t: `{val_high_t_loss:.6f}`\n')
            f.write(f'- step: `{global_step}`\n')
            if epoch % sample_every == 0:
                f.write(f'- target_gif: `{sample_dir / f"epoch_{epoch:03d}_target.gif"}`\n')
                f.write(f'- target_png: `{sample_dir / f"epoch_{epoch:03d}_target.png"}`\n')
                f.write(f'- sample_gif: `{sample_dir / f"epoch_{epoch:03d}_sample.gif"}`\n')
                f.write(f'- sample_png: `{sample_dir / f"epoch_{epoch:03d}_sample.png"}`\n')
            f.write('\n')

        if max_steps and global_step >= max_steps:
            break
