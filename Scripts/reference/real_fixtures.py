#!/usr/bin/env python3
"""Real-weight fixtures for the decoders (card T-1.5), without the gated image encoders.

The latent comes from the real denoiser run on the *unconditional* branch (cond = zeros, the
negative condition of CFG, cfg 0): a plausible object latent that needs neither DINOv3 nor the VAE.
Every draw is recorded (latent/camera noise, octree offsets per cell, in-cell jitter).

Two descents are recorded from the same latent: the reference precision (fp16 blocks, on MPS) and
fp32 end to end on the CPU (`fp32.` prefix) — the only one where anchors can be compared bit for bit:
in fp16 a tiny probability difference moves one point to the neighbouring cell from level 2 on.
The cells each level saw (`cells_<l>`) are recorded so the network can be checked level by level.

Usage: .venv-ref/bin/python -I Scripts/reference/real_fixtures.py [--steps 50] [--num-gaussians 262144]
         [--reuse-latent]   (keep the latent of an existing decoders.safetensors: skips the ~6 min denoiser)
Writes $SPERID_MODELS_DIR/parity/decoders.safetensors and uncond.ply (never committed).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OctreeNoiseRecorder, meta, save_deterministic  # noqa: E402

import idol.models.decoder as decoder_mod  # noqa: E402
from idol.models import GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder  # noqa: E402
from idol.sampler import sample_flow  # noqa: E402

SEED = 0


def unconditional_latent(load, config, generator, steps, device):
    den = load(LatentSeqMMFlowModel(**config["denoiser"]["args"]), "denoiser", device)
    latent_noise = torch.randn(1, den.q_token_length, den.in_channels, generator=generator)
    camera_noise = torch.randn(1, 1, den.cam_channels, generator=generator)
    tokens = 5 + (config["image_size"] // 16) ** 2
    cond = TensorDict({"feature1": torch.zeros(1, tokens, 1280), "feature2": torch.zeros(1, tokens, 128)},
                      batch_size=1).to(device)
    noise = TensorDict({"latent": latent_noise, "camera": camera_noise}, batch_size=1).to(device)
    start = time.time()
    sample = sample_flow(den, noise, cond=cond, neg_cond=cond, steps=steps, cfg_strength=0.0, verbose=False)
    print(f"denoiser {steps} steps on {device}: {time.time() - start:.1f} s")
    return {"latent_noise": latent_noise, "camera_noise": camera_noise,
            "latent": sample["latent"].float().cpu(), "camera": sample["camera"].float().cpu()}


def descend(octree, latent, generator, num_points, level, prefix, out):
    """Octree descent with every draw and every level's input cells recorded under `prefix`."""
    cells = []
    hook = octree.register_forward_pre_hook(lambda m, args: cells.append(args[0].detach().float().cpu()[0]))
    rec = OctreeNoiseRecorder(decoder_mod, generator)
    try:
        anchors = rec.run(octree, latent, num_points=num_points, level=level)
    finally:
        hook.remove()
    for i, (o, p, c) in enumerate(zip(rec.offsets, rec.probs, cells)):
        out[f"{prefix}offsets_{i + 1}"], out[f"{prefix}probs_{i + 1}"], out[f"{prefix}cells_{i + 1}"] = o, p.cpu(), c
    out[f"{prefix}jitter"], out[f"{prefix}anchors"] = rec.jitter, anchors.float().cpu()
    return anchors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--num-gaussians", type=int, default=262144)
    p.add_argument("--reuse-latent", action="store_true")
    a = p.parse_args()
    root = Path(os.environ.get("SPERID_MODELS_DIR") or os.path.expanduser("~/Library/Caches/models"))
    weights = root / "speridlabs" / "idol"
    out_dir = root / "parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = os.environ.get("IDOL_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    config = json.load(open(weights / "config.json"))

    def load(model, name, dev, fp32=False):
        if fp32:
            # the constructors already cast the block Linears to fp16: back to fp32 *before* loading,
            # or the F32 checkpoint would be rounded through fp16 (a fake fp32 reference)
            model.float()
            model.dtype = torch.float32
        model.load_state_dict(load_file(str(weights / f"{name}.safetensors")), strict=True)
        return model.eval().requires_grad_(False).to(dev)

    g = torch.Generator().manual_seed(SEED)
    out = {}
    previous = out_dir / "decoders.safetensors"
    with torch.no_grad():
        if a.reuse_latent and previous.exists():
            old = load_file(str(previous))
            out.update({k: old[k] for k in ("latent_noise", "camera_noise", "latent", "camera")})
            torch.randn(1, 8192, 16, generator=g)  # keep the generator where the denoiser left it
            torch.randn(1, 1, 5, generator=g)
            print("reusing the recorded latent")
        else:
            out.update(unconditional_latent(load, config, g, a.steps, device))
        latent = out["latent"]

        octree = load(OctreeProbabilityFixedlenDecoder(**config["decoder"]["args"]), "decoder", device)
        gs_dec = load(GaussianFixedlenDecoder(**config["decoder_gs"]["args"]), "decoder_gs", device)
        num_points = max(1, a.num_gaussians // gs_dec.rep_config["num_gaussians"])
        start = time.time()
        anchors = descend(octree, latent.to(device), g, num_points, config["max_voxel_level"], "", out)
        features = gs_dec(anchors, latent.to(device))
        gaussian = gs_dec.to_representation(anchors, features)[0]
        print(f"decoders ({device}, fp16 blocks): {time.time() - start:.1f} s, {anchors.shape[1]} anchors")
        out.update({"features": features.float().cpu(), "xyz": gaussian.xyz.float().cpu(),
                    "features_dc": gaussian.features_dc.reshape(-1, 3).float().cpu(),
                    "scaling": gaussian.scaling.float().cpu(), "rotation": gaussian.rotation.float().cpu(),
                    "opacity": gaussian.opacity.float().cpu()})
        gaussian.save_ply(str(out_dir / "uncond.ply"))

        octree32 = load(OctreeProbabilityFixedlenDecoder(**config["decoder"]["args"]), "decoder", "cpu", fp32=True)
        start = time.time()
        descend(octree32, latent, g, num_points, config["max_voxel_level"], "fp32.", out)
        print(f"fp32 descent (cpu): {time.time() - start:.1f} s")
        # fp32 Gaussian decoder on the reference-precision anchors: the yardstick for fp16 (how far the
        # reference's own fp16 run sits from fp32, against how far the Swift fp16 run sits)
        gs32 = load(GaussianFixedlenDecoder(**config["decoder_gs"]["args"]), "decoder_gs", "cpu", fp32=True)
        start = time.time()
        out["fp32.features"] = gs32(out["anchors"], latent).float()
        print(f"fp32 Gaussian decoder (cpu): {time.time() - start:.1f} s")
    m = meta("real_decoders", SEED)
    m.update({"device": device, "steps": str(a.steps), "num_gaussians": str(a.num_gaussians)})
    save_deterministic(out, out_dir / "decoders.safetensors", m)
    print(f"wrote {out_dir / 'decoders.safetensors'} and uncond.ply")


if __name__ == "__main__":
    main()
