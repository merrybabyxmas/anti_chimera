from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anti_chimera.data.manifest import ManifestVideoDataset


def _tensor_frame_to_pil(frame: torch.Tensor, size: int | None = None) -> Image.Image:
    frame = frame.detach().float().clamp(0, 1)
    array = (frame.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    image = Image.fromarray(array)
    if size is not None and image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return image


def _save_grid(frames: Iterable[Image.Image], path: Path, title: str = "", columns: int = 7) -> None:
    frames = list(frames)
    if not frames:
        raise ValueError("no frames to save")
    cell_w, cell_h = frames[0].size
    label_h = 26 if title else 0
    rows = math.ceil(len(frames) / columns)
    grid = Image.new("RGB", (columns * cell_w, rows * cell_h + label_h), (255, 255, 255))
    if title:
        draw = ImageDraw.Draw(grid)
        draw.text((8, 6), title, fill=(0, 0, 0))
    for i, frame in enumerate(frames):
        x = (i % columns) * cell_w
        y = (i // columns) * cell_h + label_h
        grid.paste(frame.convert("RGB"), (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_sample(config: dict, index: int, image_size: int, num_frames: int) -> tuple[str, Image.Image, list[Image.Image]]:
    data_cfg = config["data"]
    dataset = ManifestVideoDataset(
        manifest_path=data_cfg["manifest_path"],
        root_dir=data_cfg.get("root_dir", "."),
        num_frames=num_frames,
        image_size=image_size,
        max_objects=int(data_cfg.get("max_objects", 4)),
    )
    sample = dataset[index % len(dataset)]
    video = sample["video"]
    prompt = str(sample["caption"])
    frames = [_tensor_frame_to_pil(video[:, i], image_size) for i in range(video.shape[1])]
    return prompt, frames[0], frames


def _maybe_export_video(frames: list[Image.Image], path: Path, fps: int) -> str | None:
    try:
        from diffusers.utils import export_to_video

        export_to_video(frames, str(path), fps=fps)
        return str(path)
    except Exception as exc:  # pragma: no cover - best-effort artifact
        return f"skipped: {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample CogVideoX I2V with optional LoRA and save local quality artifacts.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    config = _load_config(args.config)
    model_cfg = config["model"]
    sampling_cfg = config.get("sampling", {})
    image_size = int(args.image_size or sampling_cfg.get("image_size") or config["data"]["image_size"])
    num_frames = int(args.num_frames or sampling_cfg.get("num_frames") or config["data"]["num_frames"])
    prompt, reference_image, target_frames = _load_sample(config, args.sample_index, image_size, num_frames)

    from diffusers import CogVideoXImageToVideoPipeline

    dtype = torch.bfloat16
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        torch_dtype=dtype,
    )
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if args.cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    adapter_loaded = False
    if args.checkpoint_dir is not None and not args.base_only:
        pipe.load_lora_weights(str(args.checkpoint_dir), weight_name="pytorch_lora_weights.safetensors", adapter_name="anti_chimera")
        pipe.set_adapters(["anti_chimera"], [args.lora_scale])
        adapter_loaded = True

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    result = pipe(
        image=reference_image,
        prompt=prompt,
        height=image_size,
        width=image_size,
        num_frames=num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="pil",
    )
    frames = result.frames[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = "base" if args.base_only else "lora"
    grid_path = args.output_dir / f"{tag}_sample_grid.png"
    target_path = args.output_dir / "target_grid.png"
    ref_path = args.output_dir / "reference_frame.png"
    gif_path = args.output_dir / f"{tag}_sample.gif"
    mp4_path = args.output_dir / f"{tag}_sample.mp4"

    reference_image.save(ref_path)
    _save_grid(target_frames, target_path, title="target/reference video")
    _save_grid(frames, grid_path, title=f"{tag} generated video")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0)
    mp4_status = _maybe_export_video(frames, mp4_path, fps=args.fps)

    metadata = {
        "config": str(args.config),
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "pretrained_model": model_cfg["pretrained_model_name_or_path"],
        "adapter_loaded": adapter_loaded,
        "sample_index": args.sample_index,
        "seed": args.seed,
        "image_size": image_size,
        "num_frames": num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "lora_scale": args.lora_scale,
        "prompt": prompt,
        "reference_frame": str(ref_path),
        "target_grid": str(target_path),
        "sample_grid": str(grid_path),
        "sample_gif": str(gif_path),
        "sample_mp4": mp4_status,
    }
    with open(args.output_dir / f"{tag}_sample_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
