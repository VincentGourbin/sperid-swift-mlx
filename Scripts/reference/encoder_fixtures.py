#!/usr/bin/env python3
"""Real-object fixtures for the image encoders (Jalon 3): DINOv3 ViT-H+/16 and the FLUX.2 VAE encoder.

From the avocado's preprocessed pixels (parity/denoiser_avocado.safetensors, `pixels`):
  dino32_tokens      DINOv3 fp32 (CPU) `last_hidden_state` after Idol's parameter-free layer_norm = feature1, fp32
  vae32_mean/logvar  FLUX.2 VAE encoder fp32 (CPU): the posterior moments [1, 32, 128, 128] (after quant_conv)
  vae_noise          the posterior ε [1, 32, 128, 128] (drawn here, CPU generator)
  feature1_bf16, feature2_bf16   the SDK's own ImageEncoder path (bf16, CPU) with that ε replayed
  vae32_feature2     feature2 computed from the fp32 moments and ε through the SDK's _flux2_tokens code

Usage: .venv-ref/bin/python -I Scripts/reference/encoder_fixtures.py
Writes $SPERID_MODELS_DIR/parity/encoders_avocado.safetensors (never committed).
"""
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import meta, save_deterministic  # noqa: E402

from diffusers.models import AutoencoderKLFlux2  # noqa: E402
from diffusers.models.autoencoders import vae as vae_module  # noqa: E402
from transformers import AutoModel  # noqa: E402

from idol.conditioning import ImageEncoder  # noqa: E402

DINO = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
FLUX = "black-forest-labs/FLUX.2-dev"


def main():
    root = Path(os.environ.get("SPERID_MODELS_DIR") or os.path.expanduser("~/Library/Caches/models"))
    pixels = load_file(str(root / "parity" / "denoiser_avocado.safetensors"))["pixels"].float()
    g = torch.Generator().manual_seed(0)
    eps = torch.randn(1, 32, 128, 128, generator=g)
    out = {"pixels": pixels, "vae_noise": eps}

    # replay ε in every posterior sample (the SDK calls latent_dist.sample())
    vae_module.DiagonalGaussianDistribution.sample = (
        lambda self, generator=None: self.mean + self.std * eps.to(self.mean.device, self.mean.dtype))

    with torch.no_grad():
        start = time.time()
        dino32 = AutoModel.from_pretrained(DINO, dtype=torch.float32).eval()
        vae32 = AutoencoderKLFlux2.from_pretrained(FLUX, subfolder="vae", dtype=torch.float32).eval()
        enc32 = ImageEncoder(dino32, vae32, device="cpu")
        out["dino32_tokens"] = enc32._dinov3_tokens(pixels)
        posterior = vae32.encode(pixels * 2 - 1).latent_dist
        out["vae32_mean"], out["vae32_logvar"] = posterior.mean, posterior.logvar
        # the SDK's tokens code on the fp32 VAE: _flux2_tokens casts the images to bf16, so feed it a VAE
        # wrapper that runs in fp32 — the patchify / BN / padding code itself is the SDK's
        class FP32(torch.nn.Module):
            def __init__(self, v):
                super().__init__()
                self.v, self.bn, self.config = v, v.bn, v.config

            def encode(self, x):
                return self.v.encode(x.float())
        enc32.flux2_vae = FP32(vae32)
        out["vae32_feature2"] = enc32._flux2_tokens(pixels)
        print(f"fp32 encoders (cpu): {time.time() - start:.1f} s")
        del dino32, vae32, enc32

        start = time.time()
        enc16 = ImageEncoder(DINO, FLUX, dinov3_dtype="bf16", device="cpu")
        cond = enc16(pixels)
        out["feature1_bf16"], out["feature2_bf16"] = cond["feature1"].float(), cond["feature2"].float()
        print(f"bf16 encoders (cpu): {time.time() - start:.1f} s")

    path = root / "parity" / "encoders_avocado.safetensors"
    save_deterministic(out, path, meta("encoders_avocado", 0))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
