from __future__ import annotations

import argparse
from pathlib import Path

import torch

from anti_chimera.config import load_config
from anti_chimera.data.scene_hint import SceneHintBuilder
from anti_chimera.data.synthetic_collision import SyntheticCollisionDataset
from anti_chimera.models.model import AntiChimeraVideoDiffusion
from anti_chimera.utils import default_device, normalize_video, save_gif, save_video_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--start-timestep", type=int, default=50)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = default_device(config["training"]["device"])
    data_cfg = config["data"]
    model_cfg = config["model"]

    dataset = SyntheticCollisionDataset(
        size=max(int(data_cfg.get("val_size", 1)), args.index + 1),
        num_frames=int(data_cfg["num_frames"]),
        image_size=int(data_cfg["image_size"]),
        max_objects=int(data_cfg["max_objects"]),
        seed=int(config["seed"]) + 1,
    )
    item = dataset[args.index]

    builder = SceneHintBuilder(
        max_objects=int(data_cfg["max_objects"]),
        depth_bins=int(data_cfg["depth_bins"]),
        image_size=int(data_cfg["image_size"]),
    )
    cond = builder.build(item).unsqueeze(0).to(device).float()

    cond_channels = int(data_cfg["max_objects"]) + int(data_cfg["depth_bins"]) + int(data_cfg["max_objects"])
    model = AntiChimeraVideoDiffusion(cond_channels=cond_channels, **model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = dict(ckpt.get("ema_model") or ckpt["model"])
    for key in list(state.keys()):
        if key.endswith("scheduler.timesteps"):
            state.pop(key)
    model.load_state_dict(state, strict=False)
    model.eval()

    source = normalize_video(item["video"].unsqueeze(0).to(device).float())
    t = torch.tensor([max(int(args.start_timestep), 1)], device=device, dtype=torch.long)
    noise = torch.randn_like(source)
    noisy = model.add_noise(source, noise, t)
    prediction = model(noisy, t, [item["caption"]], cond)
    refined = model.scheduler.step(prediction, t, noisy).prev_sample
    refined = ((refined[0].detach().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)

    out_path = Path(args.out)
    save_gif(refined * 2 - 1, out_path)
    save_video_png(refined * 2 - 1, out_path.with_suffix(".png"))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
