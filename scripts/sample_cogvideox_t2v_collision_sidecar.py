from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusers import CogVideoXPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_cogvideox import XLA_AVAILABLE, retrieve_timesteps
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.schedulers import CogVideoXDPMScheduler

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm

from anti_chimera.data.collision_planner import build_collision_plan
from anti_chimera.models.control import LatentControlEncoder
from anti_chimera.trainer_cogvideox import SceneConditionEncoder


class SidecarCogVideoXPipeline(CogVideoXPipeline):
    """CogVideoX T2V pipeline with minimal planned-sidecar hooks."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int = 50,
        timesteps: list[int] | None = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 226,
        sidecar_prompt_bias: torch.Tensor | None = None,
        sidecar_prompt_scale: float = 0.0,
        sidecar_latent_residual: torch.Tensor | None = None,
        sidecar_latent_scale: float = 0.0,
    ) -> CogVideoXPipelineOutput | tuple:
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames
        num_videos_per_prompt = 1

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if sidecar_prompt_bias is not None and sidecar_prompt_scale != 0.0:
            bias = sidecar_prompt_bias.to(device=device, dtype=prompt_embeds.dtype)
            if bias.ndim == 2:
                bias = bias.unsqueeze(1)
            prompt_embeds = prompt_embeds + float(sidecar_prompt_scale) * bias
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        timestep_device = "cpu" if XLA_AVAILABLE else device
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, timestep_device, timesteps
        )
        self._num_timesteps = len(timesteps)

        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        latent_residual = None
        if sidecar_latent_residual is not None and sidecar_latent_scale != 0.0:
            latent_residual = sidecar_latent_residual.to(device=device, dtype=latents.dtype)
            if latent_residual.shape[1] != latents.shape[1]:
                if latent_residual.shape[1] < latents.shape[1]:
                    pad = latent_residual.new_zeros(
                        latent_residual.shape[0],
                        latents.shape[1] - latent_residual.shape[1],
                        latent_residual.shape[2],
                        latent_residual.shape[3],
                        latent_residual.shape[4],
                    )
                    latent_residual = torch.cat([latent_residual, pad], dim=1)
                else:
                    latent_residual = latent_residual[:, : latents.shape[1]]

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if latent_residual is not None:
                    residual_input = (
                        torch.cat([latent_residual] * 2) if do_classifier_free_guidance else latent_residual
                    )
                    latent_model_input = latent_model_input + float(sidecar_latent_scale) * residual_input
                timestep = t.expand(latent_model_input.shape[0])
                with self.transformer.cache_context("cond_uncond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = noise_pred.float()
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if output_type != "latent":
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents
        self.maybe_free_model_hooks()
        if not return_dict:
            return (video,)
        return CogVideoXPipelineOutput(frames=video)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_prompt(path: Path, index: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError(f"empty prompt file: {path}")
    return rows[index % len(rows)]


def _save_grid(frames: list[Image.Image], path: Path, title: str, columns: int = 7) -> None:
    cell_w, cell_h = frames[0].size
    label_h = 26
    rows = math.ceil(len(frames) / columns)
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h + label_h), "white")
    ImageDraw.Draw(canvas).text((8, 6), title, fill=(0, 0, 0))
    for i, frame in enumerate(frames):
        canvas.paste(frame.convert("RGB"), ((i % columns) * cell_w, (i // columns) * cell_h + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _planned_frames(plan, spec: dict[str, Any], image_size: int) -> list[Image.Image]:
    rgb_a = tuple(int(x) for x in spec["rgb_a"])
    rgb_b = tuple(int(x) for x in spec["rgb_b"])
    frames: list[Image.Image] = []
    masks = plan.masks.detach().cpu().numpy()
    for t in range(masks.shape[0]):
        canvas = Image.new("RGB", (image_size, image_size), (245, 245, 245))
        arr = torch.zeros(3, image_size, image_size)
        for c, val in enumerate(rgb_a):
            arr[c][plan.masks[t, 0] > 0.1] = val / 255.0
        for c, val in enumerate(rgb_b):
            arr[c][plan.masks[t, 1] > 0.1] = torch.maximum(
                arr[c][plan.masks[t, 1] > 0.1],
                torch.full_like(arr[c][plan.masks[t, 1] > 0.1], val / 255.0),
            )
        np_img = (arr.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
        fg = Image.fromarray(np_img)
        canvas.paste(fg, mask=Image.fromarray(((masks[t, :2].max(axis=0) > 0.1) * 255).astype("uint8")))
        frames.append(canvas)
    return frames


def _load_sidecar_modules(config: dict, checkpoint_dir: Path, pipe, dtype: torch.dtype):
    data_cfg = config["data"]
    train_cfg = config["training"]
    cond_channels = int(data_cfg["max_objects"]) * 3 + int(data_cfg["depth_bins"]) + 3
    hidden_size = int(
        getattr(pipe.text_encoder.config, "hidden_size", None)
        or getattr(pipe.text_encoder.config, "d_model", 4096)
    )
    latent_channels = int(pipe.transformer.config.in_channels)
    scene_encoder = SceneConditionEncoder(
        cond_channels=cond_channels,
        hidden_size=hidden_size,
        token_count=int(train_cfg.get("scene_token_count", 4)),
    )
    latent_controller = LatentControlEncoder(
        cond_channels=cond_channels,
        latent_channels=latent_channels,
        hidden_channels=int(train_cfg.get("latent_control_hidden_channels", 256)),
        gate_init=float(train_cfg.get("latent_control_gate_init", -2.0)),
    )
    for filename, module in {
        "scene_encoder.pt": scene_encoder,
        "latent_controller.pt": latent_controller,
    }.items():
        path = checkpoint_dir / filename
        if path.exists():
            module.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        module.to(dtype=dtype)
        module.eval()
    return scene_encoder, latent_controller


def _latent_target_shape(pipe, image_size: int, num_frames: int) -> tuple[int, int, int]:
    latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    patch_size_t = pipe.transformer.config.patch_size_t
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        latent_frames += patch_size_t - latent_frames % patch_size_t
    return (
        latent_frames,
        image_size // pipe.vae_scale_factor_spatial,
        image_size // pipe.vae_scale_factor_spatial,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Text-only collision A/B sampler for CogVideoX T2V.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--scene-scale", type=float, default=None)
    parser.add_argument("--latent-scale", type=float, default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--variants", nargs="*", default=["no_sidecar", "full_sidecar"])
    args = parser.parse_args()

    config = _load_yaml(args.config)
    train_cfg = config["training"]
    dtype = torch.bfloat16 if str(train_cfg.get("mixed_precision", "fp16")).lower() == "bf16" else torch.float16
    scene_scale = float(args.scene_scale if args.scene_scale is not None else train_cfg.get("scene_prompt_scale", 0.0))
    latent_scale = float(args.latent_scale if args.latent_scale is not None else train_cfg.get("latent_control_scale", 0.0))
    spec = _load_prompt(args.prompt_file, args.prompt_index)

    pipe = SidecarCogVideoXPipeline.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        torch_dtype=dtype,
        variant=None if str(config["model"].get("variant", "none")).lower() in {"none", "null"} else config["model"].get("variant"),
    )
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if (args.checkpoint_dir / "pytorch_lora_weights.safetensors").exists():
        pipe.load_lora_weights(str(args.checkpoint_dir), weight_name="pytorch_lora_weights.safetensors", adapter_name="anti_chimera")
        pipe.set_adapters(["anti_chimera"], [args.lora_scale])
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=False)

    scene_encoder, latent_controller = _load_sidecar_modules(config, args.checkpoint_dir, pipe, dtype=torch.float32)
    plan = build_collision_plan(
        spec,
        num_frames=args.num_frames,
        image_size=args.image_size,
        max_objects=int(config["data"]["max_objects"]),
        depth_bins=int(config["data"]["depth_bins"]),
    )
    with torch.no_grad():
        prompt_bias = scene_encoder(plan.cond).unsqueeze(1).detach()
        latent_residual = latent_controller(plan.cond, _latent_target_shape(pipe, args.image_size, args.num_frames)).detach()

    variants = {
        "no_sidecar": (None, 0.0, None, 0.0),
        "full_sidecar": (prompt_bias, scene_scale, latent_residual, latent_scale),
        "latent_control": (None, 0.0, latent_residual, latent_scale),
        "prompt_bias": (prompt_bias, scene_scale, None, 0.0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_grid(_planned_frames(plan, spec, args.image_size), args.output_dir / "planned_sidecar_grid.png", "planned collision sidecar")
    torch.save(
        {
            "masks": plan.masks,
            "tracks": plan.tracks,
            "depth": plan.depth,
            "flow": plan.flow,
            "occlusion": plan.occlusion,
            "visibility": plan.visibility,
        },
        args.output_dir / "planned_sidecar.pt",
    )
    metadata = {
        "config": str(args.config),
        "checkpoint_dir": str(args.checkpoint_dir),
        "prompt_file": str(args.prompt_file),
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "image_size": args.image_size,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "scene_scale": scene_scale,
        "latent_scale": latent_scale,
        "lora_scale": args.lora_scale,
        "spec": spec,
        "variants": {},
    }
    for name in args.variants:
        prompt_bias_i, prompt_scale_i, latent_residual_i, latent_scale_i = variants[name]
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        result = pipe(
            prompt=spec["prompt"],
            height=args.image_size,
            width=args.image_size,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="pil",
            sidecar_prompt_bias=prompt_bias_i,
            sidecar_prompt_scale=prompt_scale_i,
            sidecar_latent_residual=latent_residual_i,
            sidecar_latent_scale=latent_scale_i,
        )
        frames = result.frames[0]
        png_path = args.output_dir / f"{name}_sample_grid.png"
        gif_path = args.output_dir / f"{name}_sample.gif"
        _save_grid(frames, png_path, f"{name} generated collision video")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=125, loop=0)
        metadata["variants"][name] = {"png": str(png_path), "gif": str(gif_path)}
    with (args.output_dir / "collision_sample_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
