#!/usr/bin/env python3
"""Real-object fixtures for the denoiser (Jalon 2): one gallery image through the upstream SDK.

Records, for `--image` (default the avocado of the gallery):
  pixels        [1, 3, 1024, 1024] the preprocessed image (BiRefNet + crop), for Jalon 3
  feature1/2    the real conditioning (DINOv3, FLUX.2 VAE) the SDK computes
  latent_noise, camera_noise   the start of the flow (drawn here, CPU generator, seed 0)
  v{16,32}_{cond,neg}_{latent,camera}  one denoiser forward at t = 1000 on the noise, fp16 (device) and fp32 (CPU)
  traj_{k}_latent/camera       the sample entering steps k ∈ {0, 10, 25, 49} of the 50-step CFG run
  latent, camera               its final sample (the object), and <name>_fixture.ply decoded by the SDK

Usage: .venv-ref/bin/python -I Scripts/reference/denoiser_fixtures.py [--image IMG] [--mode synthetic|real]
         [--steps 50] [--skip-fp32]
Writes $SPERID_MODELS_DIR/parity/denoiser_<name>.safetensors (never committed).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, meta, save_deterministic  # noqa: E402

from idol import IdolPipeline  # noqa: E402
from idol.models import LatentSeqMMFlowModel  # noqa: E402
from idol.sampler import sample_flow  # noqa: E402

SEED = 0
CHECKPOINTS = (0, 10, 25, 49)


def velocity(model, latent, camera, t, cond, device):
    x = TensorDict({"latent": latent, "camera": camera}, batch_size=1).to(device)
    v = model(x, torch.tensor([float(t)], device=device), cond.to(device))
    return v["latent"].float().cpu(), v["camera"].float().cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=str(ROOT / "reference/idol-sdk/assets/gallery/avocado.webp"))
    p.add_argument("--mode", default="synthetic")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--skip-fp32", action="store_true")
    a = p.parse_args()
    root = Path(os.environ.get("SPERID_MODELS_DIR") or os.path.expanduser("~/Library/Caches/models"))
    weights = root / "speridlabs" / "idol"
    name = Path(a.image).stem
    device = os.environ.get("IDOL_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    pipe = IdolPipeline.from_pretrained(str(weights), device=device)
    out = {}
    with torch.no_grad():
        prepared = pipe.preprocess(a.image, mode=a.mode)
        pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None]
        torch.manual_seed(42)
        cond = pipe.image_encoder(pixels.to(pipe.device))
        cond_cpu = cond.apply(lambda t: t.float().cpu())
        neg_cpu = cond_cpu.apply(torch.zeros_like)
        out.update({"pixels": pixels, "feature1": cond_cpu["feature1"], "feature2": cond_cpu["feature2"]})

        g = torch.Generator().manual_seed(SEED)
        den = pipe.denoiser
        latent_noise = torch.randn(1, den.q_token_length, den.in_channels, generator=g)
        camera_noise = torch.randn(1, 1, den.cam_channels, generator=g)
        out.update({"latent_noise": latent_noise, "camera_noise": camera_noise})
        for tag, c in (("cond", cond_cpu), ("neg", neg_cpu)):
            out[f"v16_{tag}_latent"], out[f"v16_{tag}_camera"] = velocity(den, latent_noise, camera_noise, 1000, c, device)
        if not a.skip_fp32:
            den32 = LatentSeqMMFlowModel(**pipe.config["denoiser"]["args"]).float()
            den32.dtype = torch.float32
            den32.load_state_dict(load_file(str(weights / "denoiser.safetensors")), strict=True)
            den32.eval()
            start = time.time()
            for tag, c in (("cond", cond_cpu), ("neg", neg_cpu)):
                out[f"v32_{tag}_latent"], out[f"v32_{tag}_camera"] = velocity(den32, latent_noise, camera_noise, 1000, c, "cpu")
            print(f"fp32 forwards (cpu): {time.time() - start:.1f} s")
            del den32

        calls = []
        hook = den.register_forward_pre_hook(lambda m, args: calls.append(
            (args[0]["latent"].float().cpu().clone(), args[0]["camera"].float().cpu().clone())))
        noise = TensorDict({"latent": latent_noise, "camera": camera_noise}, batch_size=1).to(device)
        start = time.time()
        sample = sample_flow(den, noise, cond=cond, neg_cond=cond.apply(torch.zeros_like), steps=a.steps,
                             rescale_t=3.0, cfg_strength=3.0, verbose=False, batch_cfg=False)
        print(f"{a.steps}-step CFG sampling on {device}: {time.time() - start:.1f} s")
        hook.remove()
        for k in CHECKPOINTS:
            if 2 * k < len(calls):
                out[f"traj_{k}_latent"], out[f"traj_{k}_camera"] = calls[2 * k]
        out["latent"], out["camera"] = sample["latent"].float().cpu(), sample["camera"].float().cpu()
        torch.manual_seed(42)
        gaussian = pipe.decode(sample["latent"])
        gaussian.save_ply(str(root / "parity" / f"{name}_fixture.ply"))
    m = meta(f"denoiser_{name}", SEED)
    m.update({"device": device, "steps": str(a.steps), "image": name, "mode": a.mode})
    path = root / "parity" / f"denoiser_{name}.safetensors"
    save_deterministic(out, path, m)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
