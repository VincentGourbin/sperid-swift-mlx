#!/usr/bin/env python3
"""BiRefNet fixtures (cards T-4.3 → T-4.5).

tiny (committed, parity/tiny_birefnet.safetensors): torchvision `deform_conv2d` (modulated, kernels 1/3/7,
  offsets large enough to leave the image) and `F.interpolate(bilinear, align_corners=True)`.
real (`$SPERID_MODELS_DIR/parity/birefnet_<name>.safetensors`): the upstream BiRefNet in fp32 on CPU on one
  image — the normalised 1024² input, the 4 backbone features (after the half-resolution concat), the
  squeezed x4, the alpha [0, 1] at 1024², and `remove_background`'s uint8 alpha at the image size.

Usage: .venv-ref/bin/python -I Scripts/reference/birefnet_fixtures.py [--image IMG] [--tiny-only]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision.ops import deform_conv2d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, meta, save_deterministic  # noqa: E402

from idol.models import BiRefNet  # noqa: E402

SEED = 7


def tiny():
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for k in (1, 3, 7):
        x = torch.randn(1, 6, 9, 11, generator=g)
        w = torch.randn(5, 6, k, k, generator=g) * 0.2
        off = torch.randn(1, 2 * k * k, 9, 11, generator=g) * 2.5   # pushes samples outside the image
        mask = torch.rand(1, k * k, 9, 11, generator=g) * 2
        y = deform_conv2d(x, off, w, None, padding=k // 2, mask=mask)
        out.update({f"deform{k}.x": x, f"deform{k}.w": w, f"deform{k}.offset": off, f"deform{k}.mask": mask,
                    f"deform{k}.y": y})
    x = torch.randn(1, 4, 7, 13, generator=g)
    out["resize.x"] = x
    out["resize.up"] = F.interpolate(x, size=(16, 20), mode="bilinear", align_corners=True)
    out["resize.down"] = F.interpolate(x, size=(3, 5), mode="bilinear", align_corners=True)
    out["resize.one"] = F.interpolate(x[..., :1, :1], size=(4, 6), mode="bilinear", align_corners=True)
    return out


def real(image_path, root):
    model = BiRefNet()
    model.load_state_dict(load_file(str(root / "speridlabs" / "idol" / "birefnet.safetensors")), strict=True)
    model = model.float().eval()
    image = Image.open(image_path).convert("RGB")
    out = {}
    with torch.no_grad():
        arr = np.array(image, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=model.INPUT_SIZE, mode="bilinear", align_corners=True)
        mean = torch.tensor(model._NORM_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(model._NORM_STD).view(1, 3, 1, 1)
        x = (t - mean) / std
        out["image"] = torch.from_numpy(np.array(image)).to(torch.uint8)
        out["input"] = x
        x1, x2, x3, x4 = model._forward_enc(x)
        out.update({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
        sq = model.squeeze_module(x4)
        out["squeezed"] = sq
        alpha = torch.sigmoid(model.decoder(x, x1, x2, x3, sq))
        out["alpha1024"] = alpha
        rgba = model.remove_background(image)
        out["alpha_uint8"] = torch.from_numpy(np.array(rgba.getchannel(3)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=os.path.expanduser("~/Library/Caches/models/speridlabs-idol-space/examples/food-avocado.webp"))
    p.add_argument("--tiny-only", action="store_true")
    p.add_argument("--tiny-out", default=str(ROOT / "parity"), help="folder of the committed tiny fixture")
    a = p.parse_args()
    torch.set_num_threads(4)
    path = Path(a.tiny_out) / "tiny_birefnet.safetensors"
    save_deterministic(tiny(), path, meta("tiny_birefnet", SEED))
    print(f"wrote {path}")
    if a.tiny_only:
        return
    root = Path(os.environ.get("SPERID_MODELS_DIR") or os.path.expanduser("~/Library/Caches/models"))
    name = Path(a.image).stem
    rpath = root / "parity" / f"birefnet_{name}.safetensors"
    save_deterministic(real(a.image, root), rpath, meta(f"birefnet_{name}", SEED))
    print(f"wrote {rpath}")


if __name__ == "__main__":
    main()
