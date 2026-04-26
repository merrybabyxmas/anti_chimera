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

from diffusers import CogVideoXImageToVideoPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import XLA_AVAILABLE, retrieve_timesteps
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.schedulers import CogVideoXDPMScheduler

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm

from anti_chimera.data.manifest import ManifestVideoDataset
from anti_chimera.data.scene_hint import SceneHintBuilder
from anti_chimera.models.control import LatentControlEncoder, ReferenceConditionEncoder
from anti_chimera.trainer_cogvideox import SceneConditionEncoder, _batch_to_condition, _collate_fn


class SidecarCogVideoXImageToVideoPipeline(CogVideoXImageToVideoPipeline):
    """CogVideoX I2V pipeline with minimal sidecar hooks inside the official denoising loop."""

    @torch.no_grad()
    def __call__(
        self,
        image,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int = 49,
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
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._attention_kwargs = attention_kwargs
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
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
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

        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )

        latent_channels = self.transformer.config.in_channels // 2
        latents, image_latents = self.prepare_latents(
            image,
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
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)
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

                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
                timestep = t.expand(latent_model_input.shape[0])

                with self.transformer.cache_context("cond_uncond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        ofs=ofs_emb,
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
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _tensor_frame_to_pil(frame: torch.Tensor, image_size: int) -> Image.Image:
    frame = frame.detach().float().clamp(0, 1)
    array = (frame.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array).resize((image_size, image_size), Image.Resampling.BICUBIC)


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


def _load_sample(config: dict, sample_index: int, image_size: int) -> tuple[dict, str, Image.Image, list[Image.Image]]:
    data_cfg = config["data"]
    dataset = ManifestVideoDataset(
        manifest_path=data_cfg["manifest_path"],
        root_dir=data_cfg.get("root_dir", "."),
        num_frames=int(data_cfg["num_frames"]),
        image_size=int(data_cfg["image_size"]),
        max_objects=int(data_cfg["max_objects"]),
    )
    batch = _collate_fn([dataset[sample_index % len(dataset)]])
    video = batch["video"][0]
    target_frames = [_tensor_frame_to_pil(video[:, i], image_size) for i in range(video.shape[1])]
    return batch, str(batch["caption"][0]), target_frames[0], target_frames


def _load_sidecar_modules(config: dict, checkpoint_dir: Path, pipe, dtype: torch.dtype):
    data_cfg = config["data"]
    train_cfg = config["training"]
    cond_channels = int(data_cfg["max_objects"]) * 3 + int(data_cfg["depth_bins"]) + 3
    hidden_size = int(
        getattr(pipe.text_encoder.config, "hidden_size", None)
        or getattr(pipe.text_encoder.config, "d_model", 4096)
    )
    latent_channels = int(pipe.transformer.config.in_channels // 2)
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
    reference_encoder = ReferenceConditionEncoder(latent_channels=latent_channels, hidden_size=hidden_size)
    for filename, module in {
        "scene_encoder.pt": scene_encoder,
        "latent_controller.pt": latent_controller,
        "reference_encoder.pt": reference_encoder,
    }.items():
        path = checkpoint_dir / filename
        if path.exists():
            module.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        module.to(dtype=dtype)
        module.eval()
    return scene_encoder, latent_controller, reference_encoder


def _latent_target_shape(pipe, image_size: int, num_frames: int) -> tuple[int, int, int]:
    latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    patch_size_t = pipe.transformer.config.patch_size_t
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        latent_frames += patch_size_t - latent_frames % patch_size_t
    latent_h = image_size // pipe.vae_scale_factor_spatial
    latent_w = image_size // pipe.vae_scale_factor_spatial
    return latent_frames, latent_h, latent_w


def main() -> None:
    parser = argparse.ArgumentParser(description="Official CogVideoX I2V sidecar A/B sampler.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=13)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--scene-scale", type=float, default=None)
    parser.add_argument("--latent-scale", type=float, default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--variants", nargs="*", default=["no_sidecar", "prompt_bias", "latent_control", "full_sidecar"])
    args = parser.parse_args()

    config = _load_yaml(args.config)
    train_cfg = config["training"]
    scene_scale = float(args.scene_scale if args.scene_scale is not None else train_cfg.get("scene_prompt_scale", 0.0))
    latent_scale = float(args.latent_scale if args.latent_scale is not None else train_cfg.get("latent_control_scale", 0.0))
    dtype = torch.bfloat16 if str(train_cfg.get("mixed_precision", "bf16")).lower() == "bf16" else torch.float16

    batch, prompt, reference_image, target_frames = _load_sample(config, args.sample_index, args.image_size)
    pipe = SidecarCogVideoXImageToVideoPipeline.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        torch_dtype=dtype,
    )
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.load_lora_weights(str(args.checkpoint_dir), weight_name="pytorch_lora_weights.safetensors", adapter_name="anti_chimera")
    pipe.set_adapters(["anti_chimera"], [args.lora_scale])
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=False)

    scene_encoder, latent_controller, _ = _load_sidecar_modules(config, args.checkpoint_dir, pipe, dtype=torch.float32)
    builder = SceneHintBuilder(
        max_objects=int(config["data"]["max_objects"]),
        depth_bins=int(config["data"]["depth_bins"]),
        image_size=int(config["data"]["image_size"]),
    )
    cond = _batch_to_condition(batch, config, builder, torch.device("cpu"))
    with torch.no_grad():
        prompt_bias = scene_encoder(cond).unsqueeze(1).detach()
        latent_residual = latent_controller(cond, _latent_target_shape(pipe, args.image_size, args.num_frames)).detach()

    variant_cfg = {
        "no_sidecar": (None, 0.0, None, 0.0),
        "prompt_bias": (prompt_bias, scene_scale, None, 0.0),
        "latent_control": (None, 0.0, latent_residual, latent_scale),
        "full_sidecar": (prompt_bias, scene_scale, latent_residual, latent_scale),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_grid(target_frames, args.output_dir / "target_grid.png", "target/reference video")
    reference_image.save(args.output_dir / "reference_frame.png")

    metadata = {
        "config": str(args.config),
        "checkpoint_dir": str(args.checkpoint_dir),
        "sample_index": args.sample_index,
        "seed": args.seed,
        "image_size": args.image_size,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "scene_scale": scene_scale,
        "latent_scale": latent_scale,
        "prompt": prompt,
        "variants": {},
    }
    for name in args.variants:
        prompt_bias_i, prompt_scale_i, latent_residual_i, latent_scale_i = variant_cfg[name]
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        result = pipe(
            image=reference_image,
            prompt=prompt,
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
        _save_grid(frames, png_path, f"{name} generated video")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=125, loop=0)
        metadata["variants"][name] = {"png": str(png_path), "gif": str(gif_path)}

    with open(args.output_dir / "sidecar_ablation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
