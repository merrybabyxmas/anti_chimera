from __future__ import annotations

import json
import math
import os
os.environ.setdefault('NCCL_P2P_DISABLE', '1')
os.environ.setdefault('NCCL_IB_DISABLE', '1')
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from anti_chimera.data.manifest import ManifestVideoDataset
from anti_chimera.data.scene_hint import SceneHintBuilder
from anti_chimera.inference import build_null_condition
from anti_chimera.models.control import LatentControlEncoder, ReferenceConditionEncoder
from anti_chimera.utils import default_device, ensure_dir, normalize_video, save_gif, save_video_png

logger = get_logger(__name__)


class SceneConditionEncoder(nn.Module):
    def __init__(self, cond_channels: int, hidden_size: int, token_count: int = 4) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Sequential(
            nn.LayerNorm(cond_channels),
            nn.Linear(cond_channels, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        nn.init.zeros_(self.proj[-1].weight)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        cond = cond.float()
        pooled = self.pool(cond).flatten(1).to(dtype=self.gate.dtype)
        bias = self.proj(pooled)
        return (1.0 + torch.sigmoid(self.gate)) * bias


def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
    return {
        "video": torch.stack([item["video"] for item in batch], dim=0),
        "caption": [item["caption"] for item in batch],
        "tracks": torch.stack([item["tracks"] for item in batch], dim=0),
        "depth": torch.stack([item["depth"] for item in batch], dim=0),
        "visibility": torch.stack([item["visibility"] for item in batch], dim=0),
        "masks": torch.stack([item["masks"] for item in batch], dim=0),
        "flow": torch.stack([item["flow"] for item in batch], dim=0),
        "occlusion": torch.stack([item["occlusion"] for item in batch], dim=0),
    }


def _collate_latent_fn(batch: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
    out = {
        "video": torch.stack([item["video"] for item in batch], dim=0),
        "caption": [item["caption"] for item in batch],
        "latents": torch.stack([item["latents"] for item in batch], dim=0),
        "reference_latents": torch.stack([item["reference_latents"] for item in batch], dim=0),
        "prompt_embeds": torch.stack([item["prompt_embeds"] for item in batch], dim=0),
        "scene_cond": torch.stack([item["scene_cond"] for item in batch], dim=0),
    }
    if batch[0].get("image_latents") is not None:
        out["image_latents"] = torch.stack([item["image_latents"] for item in batch], dim=0)
    return out


def _has_sidecar_signal(sample: Dict[str, torch.Tensor]) -> bool:
    for key in ('tracks', 'depth', 'visibility', 'masks', 'flow', 'occlusion'):
        value = sample.get(key)
        if value is not None and torch.is_tensor(value) and float(value.abs().sum().item()) > 0.0:
            return True
    return False


def _batch_to_condition(batch: Dict[str, torch.Tensor | List[str]], config: Dict, builder: SceneHintBuilder, device: torch.device) -> torch.Tensor:
    conds = []
    for i, prompt in enumerate(batch['caption']):
        masks = batch.get('masks')
        flow = batch.get('flow')
        occlusion = batch.get('occlusion')
        sample = {
            'video': batch['video'][i],
            'caption': prompt,
            'tracks': batch['tracks'][i],
            'depth': batch['depth'][i],
            'visibility': batch['visibility'][i],
            'masks': masks[i] if masks is not None else None,
            'flow': flow[i] if flow is not None else None,
            'occlusion': occlusion[i] if occlusion is not None else None,
        }
        if _has_sidecar_signal(sample):
            cond = builder.build(sample)
        else:
            cond = build_null_condition(prompt, config, device)
        conds.append(cond.squeeze(0))
    return torch.stack(conds, dim=0)


def _encode_reference_latent(vae: nn.Module, videos: torch.Tensor) -> torch.Tensor:
    if videos.ndim != 5:
        raise ValueError(f'expected normalized video tensor with 5 dims, got {tuple(videos.shape)}')
    reference_video = videos[:, :, :1]
    scaling_factor = float(getattr(vae.config, 'scaling_factor', 1.0))
    with torch.no_grad():
        reference_latent = vae.encode(reference_video).latent_dist.sample() * scaling_factor
    return reference_latent


def _resize_video_tensor(video: torch.Tensor, image_size: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f'expected video tensor with 5 dims, got {tuple(video.shape)}')
    if video.shape[-2:] == (image_size, image_size):
        return video
    batch, channels, frames, _, _ = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, video.shape[-2], video.shape[-1])
    flat = F.interpolate(flat, size=(image_size, image_size), mode='bilinear', align_corners=False)
    return flat.reshape(batch, frames, channels, image_size, image_size).permute(0, 2, 1, 3, 4).contiguous()


def _supports_i2v_reference(transformer_config, latent_channels: int) -> bool:
    return int(getattr(transformer_config, 'in_channels', latent_channels)) == latent_channels * 2


def _reference_mode(config: Dict) -> str:
    return str(config.get('training', {}).get('reference_conditioning', 'text_bias')).lower()


def _use_i2v_reference(config: Dict, transformer_config, latent_channels: int) -> bool:
    mode = _reference_mode(config)
    if mode in {'first_frame_i2v', 'i2v', 'image_latent'}:
        return _supports_i2v_reference(transformer_config, latent_channels)
    return False


def _encode_i2v_image_latents(
    vae: nn.Module,
    videos: torch.Tensor,
    num_latent_frames: int,
    noise_sigma_mean: float | None = None,
    noise_sigma_std: float = 0.5,
) -> torch.Tensor:
    if videos.ndim != 5:
        raise ValueError(f'expected normalized video tensor with 5 dims, got {tuple(videos.shape)}')
    image = videos[:, :, :1].clone()
    if noise_sigma_mean is not None:
        sigma = torch.normal(
            mean=float(noise_sigma_mean),
            std=float(noise_sigma_std),
            size=(image.shape[0],),
            device=image.device,
        ).exp().to(dtype=image.dtype)
        image = image + torch.randn_like(image) * sigma[:, None, None, None, None]
    scaling_factor = float(getattr(vae.config, 'scaling_factor', 1.0))
    with torch.no_grad():
        image_latents = vae.encode(image).latent_dist.sample() * scaling_factor
    image_latents = image_latents.permute(0, 2, 1, 3, 4).contiguous()
    pad_frames = max(0, int(num_latent_frames) - image_latents.shape[1])
    if pad_frames > 0:
        padding = image_latents.new_zeros(
            image_latents.shape[0],
            pad_frames,
            image_latents.shape[2],
            image_latents.shape[3],
            image_latents.shape[4],
        )
        image_latents = torch.cat([image_latents, padding], dim=1)
    return image_latents[:, :num_latent_frames]


def _scale_with_warmup(base_scale: float, step: int, warmup_steps: int) -> float:
    if base_scale == 0.0 or warmup_steps <= 0:
        return float(base_scale)
    return float(base_scale) * min(1.0, float(step + 1) / float(warmup_steps))


def _rotary_num_frames(transformer_config, latent_frames: int) -> int:
    patch_size_t = getattr(transformer_config, 'patch_size_t', None)
    if patch_size_t is None:
        return int(latent_frames)
    return max(1, int(math.ceil(float(latent_frames) / float(patch_size_t))))


def _encode_prompt(tokenizer, text_encoder, prompts: List[str], device: torch.device, dtype: torch.dtype, max_sequence_length: int = 226) -> torch.Tensor:
    encoded = tokenizer(
        prompts,
        padding='max_length',
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors='pt',
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_embeds = text_encoder(**encoded)[0]
    return prompt_embeds.to(dtype=dtype, device=device)


def _prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int,
    patch_size: int,
    attention_head_dim: int,
    device: torch.device,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
        device=device,
    )
    return freqs_cos, freqs_sin


def _sample_validation_video(
    transformer: nn.Module,
    scene_encoder: nn.Module,
    latent_controller: nn.Module | None,
    reference_encoder: nn.Module | None,
    vae,
    tokenizer,
    text_encoder,
    scheduler,
    prompt: str,
    config: Dict,
    device: torch.device,
    weight_dtype: torch.dtype,
    batch: Dict[str, torch.Tensor | List[str]] | None = None,
    conditioning_mode: str = 'prompt_bias',
    scene_prompt_scale: float = 1.0,
    latent_control_scale: float = 0.0,
    reference_scale: float = 0.0,
    i2v_reference_scale: float = 1.0,
) -> torch.Tensor:
    data_cfg = config['data']
    sampling_cfg = config['sampling']
    model_cfg = config['model']
    builder = SceneHintBuilder(
        max_objects=int(data_cfg['max_objects']),
        depth_bins=int(data_cfg['depth_bins']),
        image_size=int(data_cfg['image_size']),
    )

    if batch is not None and batch.get('caption'):
        prompt = str(batch['caption'][0])
        cond = _batch_to_condition(batch, config, builder, device)
    else:
        cond = build_null_condition(prompt, config, device)
    if cond.ndim == 4:
        cond = cond.unsqueeze(0)
    cond = cond.to(device=device, dtype=weight_dtype)

    variant = model_cfg.get('variant')
    if variant in {'2b', '5b', 'none', 'None'}:
        variant = None

    pipe = CogVideoXPipeline.from_pretrained(
        model_cfg['pretrained_model_name_or_path'],
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=CogVideoXDPMScheduler.from_config(scheduler.config),
        torch_dtype=weight_dtype,
        variant=variant,
        cache_dir=model_cfg.get('cache_dir'),
    )
    pipe.set_progress_bar_config(disable=True)
    use_model_cpu_offload = bool(sampling_cfg.get('enable_model_cpu_offload', False))
    use_sequential_cpu_offload = bool(sampling_cfg.get('enable_sequential_cpu_offload', False))
    if use_sequential_cpu_offload and hasattr(pipe, 'enable_sequential_cpu_offload'):
        pipe.enable_sequential_cpu_offload(gpu_id=device.index or 0)
    elif use_model_cpu_offload and hasattr(pipe, 'enable_model_cpu_offload'):
        pipe.enable_model_cpu_offload(gpu_id=device.index or 0)
    use_cpu_offload = use_model_cpu_offload or use_sequential_cpu_offload

    generator = torch.Generator(device=device).manual_seed(int(config.get('seed', 42)))
    num_steps = int(sampling_cfg['num_steps'])
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    height = int(sampling_cfg.get('image_size', data_cfg['image_size']))
    width = height
    num_frames = int(sampling_cfg.get('num_frames', data_cfg['num_frames']))
    latent_channels = int(getattr(pipe.vae.config, 'latent_channels', 16))
    use_i2v_reference = _use_i2v_reference(config, transformer.config, latent_channels)
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=latent_channels,
        num_frames=num_frames,
        height=height,
        width=width,
        dtype=weight_dtype,
        device=device,
        generator=generator,
    )

    if not use_cpu_offload:
        text_encoder.to(device=device, dtype=weight_dtype)
    prompt_embeds = _encode_prompt(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompts=[prompt],
        device=device,
        dtype=weight_dtype,
        max_sequence_length=int(model_cfg.get('max_text_seq_length', 226)),
    )
    negative_prompt_embeds = _encode_prompt(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompts=[''],
        device=device,
        dtype=weight_dtype,
        max_sequence_length=int(model_cfg.get('max_text_seq_length', 226)),
    )
    if not use_cpu_offload:
        text_encoder.to('cpu')
        free_memory()
    scene_bias = scene_encoder(cond).unsqueeze(1)
    reference_bias = None
    needs_vae_conditioning = (reference_encoder is not None and reference_scale > 0 and batch is not None) or use_i2v_reference
    if needs_vae_conditioning and not use_cpu_offload:
        vae.to(device=device, dtype=weight_dtype)
    if reference_encoder is not None and reference_scale > 0 and batch is not None:
        reference_video = normalize_video(batch['video'].to(device=device, dtype=weight_dtype))
        reference_latent = _encode_reference_latent(vae, reference_video).to(device=device, dtype=weight_dtype)
        reference_bias = reference_encoder(reference_latent).unsqueeze(1)
    image_latents = None
    if use_i2v_reference:
        if batch is not None:
            reference_video = normalize_video(batch['video'][:1].to(device=device, dtype=weight_dtype))
            reference_video = _resize_video_tensor(reference_video, height)
            image_latents = _encode_i2v_image_latents(vae, reference_video, latents.shape[1])
            image_latents = image_latents.to(device=device, dtype=weight_dtype) * float(i2v_reference_scale)
        else:
            image_latents = latents.new_zeros(latents.shape)
    if needs_vae_conditioning and bool(sampling_cfg.get('offload_vae_before_denoise', False)) and not use_cpu_offload:
        vae.to('cpu')
        free_memory()
    latent_residual = None
    if latent_controller is not None and latent_control_scale > 0:
        latent_residual = latent_controller(cond, (latents.shape[1], latents.shape[3], latents.shape[4]))

    guidance_scale = float(sampling_cfg.get('guidance_scale', 1.0))
    use_cfg = guidance_scale > 1.0
    old_pred_original_sample = None
    autocast_dtype = weight_dtype if weight_dtype in {torch.float16, torch.bfloat16} else torch.float32
    if not use_cpu_offload:
        transformer.to(device=device, dtype=weight_dtype)
    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=autocast_dtype):
        for i, timestep in enumerate(timesteps):
            t = timestep if torch.is_tensor(timestep) else torch.tensor(timestep, device=device)
            t_batch = t.reshape(1).long()
            latent_model_input = pipe.scheduler.scale_model_input(latents, t)
            if latent_residual is not None:
                latent_model_input = latent_model_input + latent_control_scale * latent_residual
            if image_latents is not None:
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            ofs_emb = (
                latent_model_input.new_full((latent_model_input.shape[0],), fill_value=2.0)
                if getattr(transformer.config, 'ofs_embed_dim', None) is not None
                else None
            )
            image_rotary_emb = (
                _prepare_rotary_positional_embeddings(
                    height=height,
                    width=width,
                    num_frames=_rotary_num_frames(transformer.config, latents.shape[1]),
                    vae_scale_factor_spatial=2 ** (len(pipe.vae.config.block_out_channels) - 1),
                    patch_size=int(getattr(transformer.config, 'patch_size', 2)),
                    attention_head_dim=int(getattr(transformer.config, 'attention_head_dim', 64)),
                    device=device,
                    base_height=int(getattr(transformer.config, 'sample_height', 60)) * (2 ** (len(pipe.vae.config.block_out_channels) - 1)),
                    base_width=int(getattr(transformer.config, 'sample_width', 90)) * (2 ** (len(pipe.vae.config.block_out_channels) - 1)),
                )
                if getattr(transformer.config, 'use_rotary_positional_embeddings', True)
                else None
            )
            cond_hidden = prompt_embeds + scene_prompt_scale * scene_bias
            if reference_bias is not None:
                cond_hidden = cond_hidden + reference_scale * reference_bias
            if use_cfg:
                uncond_hidden = negative_prompt_embeds + scene_prompt_scale * scene_bias
                if reference_bias is not None:
                    uncond_hidden = uncond_hidden + reference_scale * reference_bias
                uncond = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=uncond_hidden,
                    timestep=t_batch,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                cond_pred = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=cond_hidden,
                    timestep=t_batch,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                noise_pred = uncond + guidance_scale * (cond_pred - uncond)
            else:
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=cond_hidden,
                    timestep=t_batch,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
            step_out = pipe.scheduler.step(
                noise_pred,
                old_pred_original_sample,
                t,
                timesteps[i - 1] if i > 0 else None,
                latents,
                return_dict=False,
            )
            latents, old_pred_original_sample = step_out

    if bool(sampling_cfg.get('offload_transformer_before_decode', False)) and not use_cpu_offload:
        transformer.to('cpu')
        free_memory()
    if not use_cpu_offload:
        vae.to(device=device, dtype=weight_dtype)
    vae_param_dtype = next(pipe.vae.parameters()).dtype
    frames = pipe.decode_latents(latents.to(device=device, dtype=vae_param_dtype))
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim == 4 and frames.shape[0] in {1, 3}:
        video = frames
    elif frames.ndim == 4 and frames.shape[-1] in {1, 3}:
        video = frames.permute(3, 0, 1, 2)
    elif frames.ndim == 4 and frames.shape[1] in {1, 3}:
        video = frames.permute(1, 0, 2, 3)
    else:
        raise RuntimeError(f'Unexpected validation output shape: {tuple(frames.shape)}')
    return ((video.detach().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)


def train(config: Dict, resume_checkpoint: str | None = None) -> None:
    seed = int(config.get('seed', 42))
    set_seed(seed)

    out_dir = ensure_dir(config['output_dir'])
    ckpt_dir = ensure_dir(out_dir / 'checkpoints')
    sample_dir = ensure_dir(out_dir / 'samples')
    log_path = out_dir / 'metrics.jsonl'
    run_log = out_dir / 'run_log.md'

    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    sampling_cfg = config.get('sampling', {})

    mixed_precision = str(train_cfg.get('mixed_precision', 'fp16'))
    project_config = ProjectConfiguration(project_dir=str(out_dir), logging_dir=str(out_dir / 'logs'))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg.get('gradient_accumulation_steps', 1)),
        mixed_precision=None if mixed_precision == 'no' else mixed_precision,
        project_config=project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    device = accelerator.device if accelerator.device is not None else default_device(train_cfg.get('device', 'cuda'))
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    train_dataset = ManifestVideoDataset(
        manifest_path=data_cfg['manifest_path'],
        root_dir=data_cfg.get('root_dir', '.'),
        num_frames=int(data_cfg['num_frames']),
        image_size=int(data_cfg['image_size']),
        max_objects=int(data_cfg['max_objects']),
    )
    val_dataset = ManifestVideoDataset(
        manifest_path=data_cfg.get('val_manifest_path') or data_cfg['manifest_path'],
        root_dir=data_cfg.get('root_dir', '.'),
        num_frames=int(data_cfg['num_frames']),
        image_size=int(data_cfg['image_size']),
        max_objects=int(data_cfg['max_objects']),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get('train_batch_size', train_cfg.get('batch_size', 1))),
        shuffle=True,
        num_workers=int(train_cfg.get('num_workers', 0)),
        collate_fn=_collate_fn,
        pin_memory=device.type == 'cuda',
        persistent_workers=int(train_cfg.get('num_workers', 0)) > 0,
    )

    weight_dtype = torch.float16 if mixed_precision == 'fp16' else torch.bfloat16 if mixed_precision == 'bf16' else torch.float32

    variant = model_cfg.get('variant')
    if variant in {'2b', '5b', 'none', 'None'}:
        variant = None
    pipe = CogVideoXPipeline.from_pretrained(
        model_cfg['pretrained_model_name_or_path'],
        torch_dtype=weight_dtype,
        variant=variant,
        cache_dir=model_cfg.get('cache_dir'),
    )
    transformer = pipe.transformer
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    scheduler = pipe.scheduler

    if train_cfg.get('enable_slicing', True) and hasattr(vae, 'enable_slicing'):
        vae.enable_slicing()
    if train_cfg.get('enable_tiling', True) and hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
    if bool(train_cfg.get('gradient_checkpointing', True)) and hasattr(transformer, 'enable_gradient_checkpointing'):
        transformer.enable_gradient_checkpointing()

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.eval()
    vae.eval()

    cond_channels = int(data_cfg['max_objects']) * 3 + int(data_cfg['depth_bins']) + 3
    conditioning_mode = str(train_cfg.get('conditioning_mode', 'prompt_bias'))
    objective_mode = str(train_cfg.get('objective_mode', 'official_cogvideox'))
    if objective_mode != 'official_cogvideox':
        raise ValueError(f'unsupported objective_mode `{objective_mode}`; use `official_cogvideox`')
    scene_prompt_scale = float(train_cfg.get('scene_prompt_scale', 0.0 if conditioning_mode == 'latent_control' else 1.0))
    latent_control_scale = float(train_cfg.get('latent_control_scale', 0.25 if conditioning_mode in {'latent_control', 'hybrid'} else 0.0))
    latent_control_warmup_steps = int(train_cfg.get('latent_control_warmup_steps', 0))
    reference_scale = float(train_cfg.get('reference_scale', 0.15 if conditioning_mode in {'hybrid', 'prompt_bias'} else 0.0))
    latent_channels = int(getattr(vae.config, 'latent_channels', 16))
    use_i2v_reference = _use_i2v_reference(config, transformer.config, latent_channels)
    supports_i2v_reference = _supports_i2v_reference(transformer.config, latent_channels)
    precompute_latents = bool(train_cfg.get('precompute_latents', False))

    if precompute_latents:
        text_encoder.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        cache_builder = SceneHintBuilder(
            max_objects=int(data_cfg['max_objects']),
            depth_bins=int(data_cfg['depth_bins']),
            image_size=int(data_cfg['image_size']),
        )
        cached_items = []
        for idx in range(len(train_dataset)):
            item = train_dataset[idx]
            raw_batch = _collate_fn([item])
            videos = normalize_video(raw_batch['video'].to(device=device, dtype=weight_dtype))
            with torch.no_grad():
                latents_cached = vae.encode(videos).latent_dist.sample() * float(getattr(vae.config, 'scaling_factor', 1.0))
                latents_cached = latents_cached.permute(0, 2, 1, 3, 4).contiguous()
                reference_cached = _encode_reference_latent(vae, videos)
                prompt_cached = _encode_prompt(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompts=[str(item['caption'])],
                    device=device,
                    dtype=weight_dtype,
                    max_sequence_length=int(model_cfg.get('max_text_seq_length', 226)),
                )
                image_cached = (
                    _encode_i2v_image_latents(
                        vae,
                        videos,
                        latents_cached.shape[1],
                        noise_sigma_mean=float(train_cfg.get('i2v_image_noise_sigma_mean', -3.0)),
                        noise_sigma_std=float(train_cfg.get('i2v_image_noise_sigma_std', 0.5)),
                    ).to(device=device, dtype=weight_dtype)
                    if use_i2v_reference
                    else None
                )
                scene_cached = _batch_to_condition(raw_batch, config, cache_builder, device)
            cached_items.append({
                'video': item['video'].cpu(),
                'caption': str(item['caption']),
                'latents': latents_cached[0].detach().cpu(),
                'reference_latents': reference_cached[0].detach().cpu(),
                'prompt_embeds': prompt_cached[0].detach().cpu(),
                'scene_cond': scene_cached[0].detach().cpu(),
                'image_latents': image_cached[0].detach().cpu() if image_cached is not None else None,
            })
        train_loader = DataLoader(
            cached_items,
            batch_size=int(train_cfg.get('train_batch_size', train_cfg.get('batch_size', 1))),
            shuffle=True,
            num_workers=0,
            collate_fn=_collate_latent_fn,
            pin_memory=device.type == 'cuda',
        )
        text_encoder.to('cpu')
        vae.to('cpu')
        free_memory()
        transformer.to(device, dtype=weight_dtype)
    else:
        transformer.to(device, dtype=weight_dtype)
        text_encoder.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)

    lora_rank = int(model_cfg.get('rank', train_cfg.get('rank', 8)))
    lora_alpha = int(model_cfg.get('lora_alpha', train_cfg.get('lora_alpha', lora_rank)))
    transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True,
        target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
    )
    transformer.add_adapter(transformer_lora_config)
    scene_encoder = SceneConditionEncoder(
        cond_channels=cond_channels,
        hidden_size=int(getattr(text_encoder.config, 'hidden_size', None) or getattr(text_encoder.config, 'd_model', 4096)),
        token_count=int(train_cfg.get('scene_token_count', 4)),
    ).to(device=device, dtype=weight_dtype)
    latent_controller = LatentControlEncoder(
        cond_channels=cond_channels,
        latent_channels=latent_channels,
        hidden_channels=int(train_cfg.get('latent_control_hidden_channels', 256)),
        gate_init=float(train_cfg.get('latent_control_gate_init', -2.0)),
    ).to(device=device, dtype=weight_dtype)
    reference_encoder = ReferenceConditionEncoder(
        latent_channels=latent_channels,
        hidden_size=int(getattr(text_encoder.config, 'hidden_size', None) or getattr(text_encoder.config, 'd_model', 4096)),
        hidden_channels=int(train_cfg.get('reference_hidden_channels', 256)),
        gate_init=float(train_cfg.get('reference_gate_init', -2.0)),
    ).to(device=device, dtype=weight_dtype)

    if weight_dtype == torch.float16:
        cast_training_params([transformer, scene_encoder, latent_controller, reference_encoder], dtype=torch.float32)

    lr = float(train_cfg.get('learning_rate', train_cfg.get('lr', 1e-4)))
    optimizer = torch.optim.AdamW(
        [
            {'params': list(filter(lambda p: p.requires_grad, transformer.parameters())), 'lr': lr},
            {'params': list(filter(lambda p: p.requires_grad, scene_encoder.parameters())), 'lr': lr},
            {'params': list(filter(lambda p: p.requires_grad, latent_controller.parameters())), 'lr': lr},
            {'params': list(filter(lambda p: p.requires_grad, reference_encoder.parameters())), 'lr': lr},
        ],
        betas=(float(train_cfg.get('adam_beta1', 0.9)), float(train_cfg.get('adam_beta2', 0.95))),
        weight_decay=float(train_cfg.get('weight_decay', train_cfg.get('adam_weight_decay', 1e-4))),
        eps=float(train_cfg.get('adam_epsilon', 1e-8)),
    )

    total_train_steps = int(train_cfg.get('max_train_steps') or train_cfg.get('max_steps') or 0)
    if total_train_steps <= 0:
        epochs = int(train_cfg.get('epochs', train_cfg.get('num_train_epochs', 1)))
        total_train_steps = epochs * max(1, math.ceil(len(train_loader) / int(train_cfg.get('gradient_accumulation_steps', 1))))

    lr_scheduler = get_scheduler(
        name=str(train_cfg.get('lr_scheduler', 'constant')),
        optimizer=optimizer,
        num_warmup_steps=int(train_cfg.get('lr_warmup_steps', 0)),
        num_training_steps=total_train_steps,
    )

    transformer, scene_encoder, latent_controller, reference_encoder, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        transformer, scene_encoder, latent_controller, reference_encoder, optimizer, train_loader, lr_scheduler
    )

    transformer_base = accelerator.unwrap_model(transformer)
    scene_encoder_base = accelerator.unwrap_model(scene_encoder)
    latent_controller_base = accelerator.unwrap_model(latent_controller)
    reference_encoder_base = accelerator.unwrap_model(reference_encoder)

    def _save_hook(models, weights, output_dir):
        transformer_model = None
        scene_encoder_model = None
        latent_controller_model = None
        reference_encoder_model = None
        while models:
            model = models.pop()
            if isinstance(model, type(accelerator.unwrap_model(transformer))):
                transformer_model = model
            elif isinstance(model, type(accelerator.unwrap_model(scene_encoder))):
                scene_encoder_model = model
            elif isinstance(model, type(accelerator.unwrap_model(latent_controller))):
                latent_controller_model = model
            elif isinstance(model, type(accelerator.unwrap_model(reference_encoder))):
                reference_encoder_model = model
            elif isinstance(model, type(accelerator.unwrap_model(reference_encoder))):
                reference_encoder_model = model
            weights.pop()

        if transformer_model is not None:
            transformer_lora_layers = get_peft_model_state_dict(accelerator.unwrap_model(transformer_model))
            CogVideoXPipeline.save_lora_weights(output_dir, transformer_lora_layers=transformer_lora_layers)
        if scene_encoder_model is not None:
            torch.save(accelerator.unwrap_model(scene_encoder_model).state_dict(), os.path.join(output_dir, 'scene_encoder.pt'))
        if latent_controller_model is not None:
            torch.save(accelerator.unwrap_model(latent_controller_model).state_dict(), os.path.join(output_dir, 'latent_controller.pt'))
        if reference_encoder_model is not None:
            torch.save(accelerator.unwrap_model(reference_encoder_model).state_dict(), os.path.join(output_dir, 'reference_encoder.pt'))

    def _load_hook(models, input_dir):
        transformer_model = None
        scene_encoder_model = None
        latent_controller_model = None
        reference_encoder_model = None
        while models:
            model = models.pop()
            if isinstance(model, type(accelerator.unwrap_model(transformer))):
                transformer_model = model
            elif isinstance(model, type(accelerator.unwrap_model(scene_encoder))):
                scene_encoder_model = model
            elif isinstance(model, type(accelerator.unwrap_model(latent_controller))):
                latent_controller_model = model
            elif isinstance(model, type(accelerator.unwrap_model(reference_encoder))):
                reference_encoder_model = model

        lora_state_dict = CogVideoXPipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith('transformer.')
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        if transformer_model is not None:
            set_peft_model_state_dict(accelerator.unwrap_model(transformer_model), transformer_state_dict, adapter_name='default')

        scene_path = os.path.join(input_dir, 'scene_encoder.pt')
        if scene_encoder_model is not None and os.path.exists(scene_path):
            accelerator.unwrap_model(scene_encoder_model).load_state_dict(torch.load(scene_path, map_location='cpu'), strict=True)
        latent_path = os.path.join(input_dir, 'latent_controller.pt')
        if latent_controller_model is not None and os.path.exists(latent_path):
            accelerator.unwrap_model(latent_controller_model).load_state_dict(torch.load(latent_path, map_location='cpu'), strict=True)
        reference_path = os.path.join(input_dir, 'reference_encoder.pt')
        if reference_encoder_model is not None and os.path.exists(reference_path):
            accelerator.unwrap_model(reference_encoder_model).load_state_dict(torch.load(reference_path, map_location='cpu'), strict=True)

    accelerator.register_save_state_pre_hook(_save_hook)
    accelerator.register_load_state_pre_hook(_load_hook)


    def _load_partial_state(module: nn.Module, state_path: Path) -> None:
        state = torch.load(state_path, map_location="cpu")
        current = module.state_dict()
        filtered = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
        module.load_state_dict(filtered, strict=False)
        if accelerator.is_main_process:
            skipped = sorted(k for k in state.keys() if k not in filtered)
            if skipped:
                logger.info(f"Skipped {len(skipped)} mismatched tensors when loading {state_path.name}: {skipped[:6]}")

    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if (resume_path / "optimizer.bin").exists():
            accelerator.load_state(str(resume_path))
        else:
            if (resume_path / "pytorch_lora_weights.safetensors").exists() or (resume_path / "pytorch_lora_weights.bin").exists():
                lora_state_dict = CogVideoXPipeline.lora_state_dict(str(resume_path))
                transformer_state_dict = {
                    f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith('transformer.')
                }
                transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
                set_peft_model_state_dict(accelerator.unwrap_model(transformer), transformer_state_dict, adapter_name="default")
            scene_path = resume_path / "scene_encoder.pt"
            if scene_path.exists():
                _load_partial_state(accelerator.unwrap_model(scene_encoder), scene_path)
            latent_path = resume_path / "latent_controller.pt"
            if latent_path.exists():
                _load_partial_state(accelerator.unwrap_model(latent_controller), latent_path)
            reference_path = resume_path / "reference_encoder.pt"
            if reference_path.exists():
                _load_partial_state(accelerator.unwrap_model(reference_encoder), reference_path)

    checkpointing_steps = int(train_cfg.get('checkpointing_steps', 0) or 0)
    sample_every = int(train_cfg.get('sample_every', 1))
    grad_clip = float(train_cfg.get('grad_clip', train_cfg.get('max_grad_norm', 1.0)))
    condition_dropout_prob = float(train_cfg.get('condition_dropout_prob', 0.0))
    prompt_dropout_prob = float(train_cfg.get('prompt_dropout_prob', 0.0))
    validation_prompt = str(train_cfg.get('validation_prompt') or (val_dataset[0]['caption'] if len(val_dataset) > 0 else ''))
    builder = SceneHintBuilder(
        max_objects=int(data_cfg['max_objects']),
        depth_bins=int(data_cfg['depth_bins']),
        image_size=int(data_cfg['image_size']),
    )

    if accelerator.is_main_process:
        with open(run_log, 'w', encoding='utf-8') as f:
            print('# anti_chimera cogvideox run log', file=f)
            print('', file=f)
            print(f'- backend: `cogvideox-official`', file=f)
            print(f'- device: `{device}`', file=f)
            print(f'- output_dir: `{out_dir}`', file=f)
            print(f'- train_size: `{len(train_dataset)}`', file=f)
            print(f'- val_size: `{len(val_dataset)}`', file=f)
            print(f'- batch_size: `{int(train_cfg.get("train_batch_size", train_cfg.get("batch_size", 1)))}`', file=f)
            print(f'- max_steps: `{total_train_steps}`', file=f)
            print(f'- mixed_precision: `{mixed_precision}`', file=f)
            print(f'- lora_rank: `{lora_rank}`', file=f)
            print(f'- lora_alpha: `{lora_alpha}`', file=f)
            print(f'- objective_mode: `{objective_mode}`', file=f)
            print(f'- conditioning_mode: `{conditioning_mode}`', file=f)
            print(f'- reference_conditioning: `{_reference_mode(config)}`', file=f)
            print(f'- supports_i2v_reference: `{supports_i2v_reference}`', file=f)
            print(f'- use_i2v_reference: `{use_i2v_reference}`', file=f)
            print(f'- precompute_latents: `{precompute_latents}`', file=f)
            print(f'- scene_prompt_scale: `{scene_prompt_scale}`', file=f)
            print(f'- latent_control_scale: `{latent_control_scale}`', file=f)
            print(f'- latent_control_warmup_steps: `{latent_control_warmup_steps}`', file=f)
            print(f'- reference_scale: `{reference_scale}`', file=f)
            print(f'- scene_token_count: `{int(train_cfg.get("scene_token_count", 4))}`', file=f)
            print(f'- validation_prompt: `{validation_prompt}`', file=f)

    progress = tqdm(range(total_train_steps), disable=not accelerator.is_local_main_process, desc='steps')
    global_step = 0
    epoch = 0

    if len(train_loader) == 0:
        raise RuntimeError("train_loader is empty; cannot run training.")

    while global_step < total_train_steps:

        transformer.train()
        scene_encoder.train()
        latent_controller.train()
        reference_encoder.train()
        for batch in train_loader:
            with accelerator.accumulate([transformer, scene_encoder, latent_controller, reference_encoder]):
                cached_batch = 'latents' in batch
                if cached_batch:
                    latents = batch['latents'].to(device=device, dtype=weight_dtype)
                    reference_latents = batch['reference_latents'].to(device=device, dtype=weight_dtype)
                    prompt_embeds = batch['prompt_embeds'].to(device=device, dtype=weight_dtype)
                    scene_cond = batch['scene_cond'].to(device=device, dtype=weight_dtype)
                else:
                    videos = normalize_video(batch['video'].to(device=device, dtype=weight_dtype))
                    prompts = ['' if (prompt_dropout_prob > 0 and random.random() < prompt_dropout_prob) else prompt for prompt in batch['caption']]
                    with torch.no_grad():
                        latents = vae.encode(videos).latent_dist.sample() * float(getattr(vae.config, 'scaling_factor', 1.0))
                        latents = latents.permute(0, 2, 1, 3, 4).contiguous()
                        reference_latents = _encode_reference_latent(vae, videos)

                    prompt_embeds = _encode_prompt(
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        prompts=prompts,
                        device=device,
                        dtype=weight_dtype,
                        max_sequence_length=int(model_cfg.get('max_text_seq_length', 226)),
                    )
                    scene_cond = _batch_to_condition(batch, config, builder, device)
                if condition_dropout_prob > 0:
                    drop_mask = torch.rand(scene_cond.shape[0], device=device) < condition_dropout_prob
                    if drop_mask.any():
                        scene_cond = scene_cond.clone()
                        scene_cond[drop_mask] = 0
                scene_bias = scene_encoder(scene_cond).unsqueeze(1)
                reference_bias = reference_encoder(reference_latents).unsqueeze(1)
                encoder_hidden_states = prompt_embeds + scene_prompt_scale * scene_bias + reference_scale * reference_bias

                batch_size, num_frames, num_channels, height, width = latents.shape
                latent_residual = latent_controller(scene_cond, (num_frames, height, width))

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
                noisy_video_latents = scheduler.add_noise(latents, noise, timesteps)
                effective_latent_control_scale = _scale_with_warmup(
                    latent_control_scale,
                    global_step,
                    latent_control_warmup_steps,
                )
                latent_model_latents = noisy_video_latents + effective_latent_control_scale * latent_residual
                if use_i2v_reference:
                    if cached_batch and 'image_latents' in batch:
                        image_latents = batch['image_latents'].to(device=device, dtype=weight_dtype)
                    else:
                        image_latents = _encode_i2v_image_latents(
                            vae,
                            videos,
                            num_frames,
                            noise_sigma_mean=float(train_cfg.get('i2v_image_noise_sigma_mean', -3.0)),
                            noise_sigma_std=float(train_cfg.get('i2v_image_noise_sigma_std', 0.5)),
                        ).to(device=device, dtype=weight_dtype)
                    noisy_model_input = torch.cat([latent_model_latents, image_latents], dim=2)
                else:
                    noisy_model_input = latent_model_latents
                ofs_emb = (
                    noisy_model_input.new_full((batch_size,), fill_value=2.0)
                    if getattr(transformer_base.config, 'ofs_embed_dim', None) is not None
                    else None
                )
                image_rotary_emb = (
                    _prepare_rotary_positional_embeddings(
                        height=int(data_cfg['image_size']),
                        width=int(data_cfg['image_size']),
                        num_frames=_rotary_num_frames(transformer_base.config, num_frames),
                        vae_scale_factor_spatial=2 ** (len(vae.config.block_out_channels) - 1),
                        patch_size=int(getattr(transformer_base.config, 'patch_size', 2)),
                        attention_head_dim=int(getattr(transformer_base.config, 'attention_head_dim', 64)),
                        device=device,
                        base_height=int(getattr(transformer_base.config, 'sample_height', 60)) * (2 ** (len(vae.config.block_out_channels) - 1)),
                        base_width=int(getattr(transformer_base.config, 'sample_width', 90)) * (2 ** (len(vae.config.block_out_channels) - 1)),
                    )
                    if getattr(transformer_base.config, 'use_rotary_positional_embeddings', True)
                    else None
                )
                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)
                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1.0 / (1.0 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)
                target = latents
                loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1).mean()

                accelerator.backward(loss)
                if grad_clip > 0:
                    accelerator.clip_grad_norm_(list(transformer.parameters()) + list(scene_encoder.parameters()) + list(latent_controller.parameters()) + list(reference_encoder.parameters()), grad_clip)
                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                logs = {
                    'loss': float(loss.detach().item()),
                    'lr': float(lr_scheduler.get_last_lr()[0]),
                    'step': global_step,
                    'latent_control_scale': float(_scale_with_warmup(latent_control_scale, global_step, latent_control_warmup_steps)),
                }
                if accelerator.is_main_process:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        print(json.dumps(logs, ensure_ascii=False), file=f)
                progress.set_postfix(loss=f"{logs['loss']:.4f}", lr=f"{logs['lr']:.2e}")

                if checkpointing_steps > 0 and global_step % checkpointing_steps == 0:
                    ckpt_path = ckpt_dir / f'checkpoint-{global_step}'
                    accelerator.save_state(str(ckpt_path))

                if sample_every > 0 and global_step % sample_every == 0 and accelerator.is_main_process:
                    transformer_unwrapped = accelerator.unwrap_model(transformer)
                    scene_unwrapped = accelerator.unwrap_model(scene_encoder)
                    latent_controller_unwrapped = accelerator.unwrap_model(latent_controller)
                    reference_encoder_unwrapped = accelerator.unwrap_model(reference_encoder)
                    sample_batch = {
                        'video': batch['video'][:1].detach().cpu(),
                        'caption': batch['caption'][:1],
                        'tracks': batch['tracks'][:1].detach().cpu(),
                        'depth': batch['depth'][:1].detach().cpu(),
                        'visibility': batch['visibility'][:1].detach().cpu(),
                        'masks': batch['masks'][:1].detach().cpu() if 'masks' in batch else None,
                        'flow': batch['flow'][:1].detach().cpu() if 'flow' in batch else None,
                        'occlusion': batch['occlusion'][:1].detach().cpu() if 'occlusion' in batch else None,
                    }
                    sample = _sample_validation_video(
                        transformer=transformer_unwrapped,
                        scene_encoder=scene_unwrapped,
                        latent_controller=latent_controller_unwrapped,
                        reference_encoder=accelerator.unwrap_model(reference_encoder),
                        vae=vae,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        scheduler=scheduler,
                        prompt=validation_prompt or batch['caption'][0],
                        config=config,
                        device=device,
                        weight_dtype=weight_dtype,
                        batch=sample_batch,
                        conditioning_mode=conditioning_mode,
                        scene_prompt_scale=scene_prompt_scale,
                        latent_control_scale=latent_control_scale,
                        reference_scale=reference_scale,
                    )
                    save_video_png(sample_batch['video'][0] * 2.0 - 1.0, sample_dir / f'step_{global_step:04d}_target.png')
                    save_gif(sample * 2.0 - 1.0, sample_dir / f'step_{global_step:04d}.gif', fps=6)
                    save_video_png(sample * 2.0 - 1.0, sample_dir / f'step_{global_step:04d}.png')

                    ablation_variants = sampling_cfg.get('ablation_variants', [])
                    if ablation_variants:
                        variant_settings = {
                            'no_condition': (0.0, 0.0, 0.0, 0.0),
                            'prompt_bias_only': (scene_prompt_scale, 0.0, reference_scale, 1.0),
                            'latent_control': (0.0, latent_control_scale, reference_scale, 1.0),
                            'full_sidecar': (scene_prompt_scale, latent_control_scale, reference_scale, 1.0),
                        }
                        for variant_name in ablation_variants:
                            if variant_name not in variant_settings:
                                continue
                            scene_s, latent_s, ref_s, i2v_s = variant_settings[variant_name]
                            variant_sample = _sample_validation_video(
                                transformer=transformer_unwrapped,
                                scene_encoder=scene_unwrapped,
                                latent_controller=latent_controller_unwrapped,
                                reference_encoder=accelerator.unwrap_model(reference_encoder),
                                vae=vae,
                                tokenizer=tokenizer,
                                text_encoder=text_encoder,
                                scheduler=scheduler,
                                prompt=validation_prompt or batch['caption'][0],
                                config=config,
                                device=device,
                                weight_dtype=weight_dtype,
                                batch=sample_batch,
                                conditioning_mode=conditioning_mode,
                                scene_prompt_scale=scene_s,
                                latent_control_scale=latent_s,
                                reference_scale=ref_s,
                                i2v_reference_scale=i2v_s,
                            )
                            save_video_png(
                                variant_sample * 2.0 - 1.0,
                                sample_dir / f'step_{global_step:04d}_{variant_name}.png',
                            )

                if global_step >= total_train_steps:
                    break
        epoch += 1
        if global_step >= total_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer_unwrapped = accelerator.unwrap_model(transformer)
        scene_unwrapped = accelerator.unwrap_model(scene_encoder)
        latent_controller_unwrapped = accelerator.unwrap_model(latent_controller)
        reference_encoder_unwrapped = accelerator.unwrap_model(reference_encoder)
        final_dir = out_dir / 'final'
        final_dir.mkdir(parents=True, exist_ok=True)
        CogVideoXPipeline.save_lora_weights(
            save_directory=str(final_dir),
            transformer_lora_layers=get_peft_model_state_dict(transformer_unwrapped),
        )
        torch.save(scene_unwrapped.state_dict(), final_dir / 'scene_encoder.pt')
        torch.save(latent_controller_unwrapped.state_dict(), final_dir / 'latent_controller.pt')
        torch.save(reference_encoder_unwrapped.state_dict(), final_dir / 'reference_encoder.pt')
        with open(run_log, 'a', encoding='utf-8') as f:
            print('', file=f)
            print(f'- final_checkpoint: `{final_dir}`', file=f)
            print(f'- total_steps: `{global_step}`', file=f)
            print(f'- sample_dir: `{sample_dir}`', file=f)

    free_memory()
