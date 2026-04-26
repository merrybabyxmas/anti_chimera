from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anti_chimera.data.manifest import ManifestVideoDataset
from anti_chimera.models.control import LatentControlEncoder, ReferenceConditionEncoder
from anti_chimera.trainer_cogvideox import SceneConditionEncoder, _collate_fn, _sample_validation_video
from anti_chimera.utils import normalize_video, save_gif, save_video_png


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _prepare_batch(config: dict, sample_index: int) -> dict:
    data_cfg = config["data"]
    dataset = ManifestVideoDataset(
        manifest_path=data_cfg["manifest_path"],
        root_dir=data_cfg.get("root_dir", "."),
        num_frames=int(data_cfg["num_frames"]),
        image_size=int(data_cfg["image_size"]),
        max_objects=int(data_cfg["max_objects"]),
    )
    return _collate_fn([dataset[sample_index % len(dataset)]])


def _load_sidecar_modules(config: dict, checkpoint_dir: Path, pipe, device: torch.device, dtype: torch.dtype):
    data_cfg = config["data"]
    train_cfg = config["training"]
    cond_channels = int(data_cfg["max_objects"]) * 3 + int(data_cfg["depth_bins"]) + 3
    hidden_size = int(
        getattr(pipe.text_encoder.config, "hidden_size", None)
        or getattr(pipe.text_encoder.config, "d_model", 4096)
    )
    latent_channels = int(getattr(pipe.vae.config, "latent_channels", 16))

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
    reference_encoder = ReferenceConditionEncoder(
        latent_channels=latent_channels,
        hidden_size=hidden_size,
        hidden_channels=int(train_cfg.get("reference_hidden_channels", 256)),
        gate_init=float(train_cfg.get("reference_gate_init", -2.0)),
    )

    module_map = {
        "scene_encoder.pt": scene_encoder,
        "latent_controller.pt": latent_controller,
        "reference_encoder.pt": reference_encoder,
    }
    for filename, module in module_map.items():
        path = checkpoint_dir / filename
        if path.exists():
            module.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        module.to(device=device, dtype=dtype)
        module.eval()
    return scene_encoder, latent_controller, reference_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sidecar/control ablations with the existing CogVideoX components.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    args = parser.parse_args()

    config = _load_yaml(args.config)
    config["seed"] = args.seed
    sampling_cfg = config.setdefault("sampling", {})
    if args.image_size is not None:
        sampling_cfg["image_size"] = args.image_size
    if args.num_frames is not None:
        sampling_cfg["num_frames"] = args.num_frames
    if args.num_steps is not None:
        sampling_cfg["num_steps"] = args.num_steps
    if args.guidance_scale is not None:
        sampling_cfg["guidance_scale"] = args.guidance_scale
    sampling_cfg["enable_sequential_cpu_offload"] = True

    from diffusers import CogVideoXPipeline

    dtype = torch.bfloat16 if str(config["training"].get("mixed_precision", "bf16")) == "bf16" else torch.float16
    device = torch.device("cuda")
    pipe = CogVideoXPipeline.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        torch_dtype=dtype,
        variant=None if str(config["model"].get("variant", "none")).lower() in {"none", "null"} else config["model"].get("variant"),
    )
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.load_lora_weights(str(args.checkpoint_dir), weight_name="pytorch_lora_weights.safetensors", adapter_name="anti_chimera")
    pipe.set_adapters(["anti_chimera"], [args.lora_scale])
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()

    scene_encoder, latent_controller, reference_encoder = _load_sidecar_modules(
        config,
        args.checkpoint_dir,
        pipe,
        device,
        dtype,
    )
    batch = _prepare_batch(config, args.sample_index)

    train_cfg = config["training"]
    scene_scale = float(train_cfg.get("scene_prompt_scale", 0.0))
    latent_scale = float(train_cfg.get("latent_control_scale", 0.0))
    reference_scale = float(train_cfg.get("reference_scale", 0.0))
    variants = {
        "no_condition": (0.0, 0.0, 0.0, 0.0),
        "reference_only": (0.0, 0.0, 0.0, 1.0),
        "prompt_bias_only": (scene_scale, 0.0, reference_scale, 1.0),
        "latent_control": (0.0, latent_scale, reference_scale, 1.0),
        "full_sidecar": (scene_scale, latent_scale, reference_scale, 1.0),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_video_png(normalize_video(batch["video"])[0], args.output_dir / "target_grid.png")
    metadata = {
        "config": str(args.config),
        "checkpoint_dir": str(args.checkpoint_dir),
        "sample_index": args.sample_index,
        "seed": args.seed,
        "variants": {},
    }
    for name, (scene_s, latent_s, ref_s, i2v_s) in variants.items():
        sample = _sample_validation_video(
            transformer=pipe.transformer,
            scene_encoder=scene_encoder,
            latent_controller=latent_controller,
            reference_encoder=reference_encoder,
            vae=pipe.vae,
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            scheduler=pipe.scheduler,
            prompt=str(batch["caption"][0]),
            config=config,
            device=device,
            weight_dtype=dtype,
            batch=batch,
            conditioning_mode=str(train_cfg.get("conditioning_mode", "hybrid")),
            scene_prompt_scale=scene_s,
            latent_control_scale=latent_s,
            reference_scale=ref_s,
            i2v_reference_scale=i2v_s,
        )
        png_path = args.output_dir / f"{name}.png"
        gif_path = args.output_dir / f"{name}.gif"
        save_video_png(sample * 2.0 - 1.0, png_path)
        save_gif(sample * 2.0 - 1.0, gif_path, fps=6)
        metadata["variants"][name] = {"png": str(png_path), "gif": str(gif_path)}

    with open(args.output_dir / "sidecar_ablation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
