#!/usr/bin/env python3
"""Run the upstream Idol SDK on one image: writes <name>.ply, <name>_input.png, <name>_camera.json.

Usage: .venv-ref/bin/python -I Scripts/reference/run_sdk.py --image IMG [--mode synthetic|real]
         [--seed 42] [--steps 50] [--cfg 3.0] [--num-gaussians 262144] [--out DIR]
         [--weights DIR|REPO] [--dinov3 DIR|REPO]
Weights default to $SPERID_MODELS_DIR/speridlabs/idol when present, else the Hub repo.
Device: IDOL_DEVICE, else mps, else cpu. The SDK refuses to start without access to the gated
DINOv3 repo unless --dinov3 points at a local copy.
"""
import argparse
import json
import os
import time


def models_dir():
    return os.environ.get("SPERID_MODELS_DIR") or os.path.expanduser("~/Library/Caches/models")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--num-gaussians", type=int, default=262144)
    p.add_argument("--out", default=".local-runs/sdk")
    p.add_argument("--weights")
    p.add_argument("--dinov3")
    a = p.parse_args()

    import torch
    from idol import IdolPipeline

    local = os.path.join(models_dir(), "speridlabs", "idol")
    weights = a.weights or (local if os.path.isfile(os.path.join(local, "denoiser.safetensors")) else "speridlabs/idol")
    device = os.environ.get("IDOL_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    encoders = {"dinov3": a.dinov3} if a.dinov3 else None
    pipe = IdolPipeline.from_pretrained(weights, device=device, image_encoders=encoders)

    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.image))[0]
    start = time.time()
    out = pipe(a.image, mode=a.mode, seed=a.seed, steps=a.steps, cfg_strength=a.cfg,
               num_gaussians=a.num_gaussians)
    elapsed = time.time() - start
    out.gaussian.save_ply(os.path.join(a.out, f"{name}.ply"))
    out.preprocessed_image.save(os.path.join(a.out, f"{name}_input.png"))
    with open(os.path.join(a.out, f"{name}_camera.json"), "w") as f:
        json.dump(out.camera, f, indent=2)
    print(f"SDK OK {name} device={device} steps={a.steps} seconds={elapsed:.1f}")


if __name__ == "__main__":
    main()
