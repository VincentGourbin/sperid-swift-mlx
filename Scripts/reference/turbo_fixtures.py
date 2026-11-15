#!/usr/bin/env python3
"""idol-turbo fixtures (Jalon 5), with the repo's own turbo.py (`_inject`, `_set_active`, `turbo_sample_flow`).

From the avocado's conditioning and noise (parity/denoiser_avocado.safetensors):
  v32_{slow,fast}_latent/camera   one forward fp32 (CPU), slow expert at t = 1000, fast at t = 400
  v16_{slow,fast}_latent/camera   the same on MPS (reference precision)
  turbo_latent, turbo_camera      the 5-step guidance-free turbo sample (MPS), and avocado_turbo.ply

Usage: .venv-ref/bin/python -I Scripts/reference/turbo_fixtures.py
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import meta, save_deterministic  # noqa: E402

from idol.models import GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder  # noqa: E402
from idol.models.decoder import OctreeProbabilityFixedlenDecoder as Octree  # noqa: E402

REPO, REV = "speridlabs/idol-turbo", "45cdd41f1f4d6d70496daa7d619f57ad79307a37"


def main():
    root = Path(os.environ.get("SPERID_MODELS_DIR") or os.path.expanduser("~/Library/Caches/models"))
    weights = root / "speridlabs" / "idol"
    config = json.load(open(weights / "config.json"))
    spec = importlib.util.spec_from_file_location("turbo", hf_hub_download(REPO, "turbo.py", revision=REV))
    turbo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(turbo)
    lora = load_file(str(root / "speridlabs" / "idol-turbo" / "idol_turbo_lora.safetensors"))
    f = load_file(str(root / "parity" / "denoiser_avocado.safetensors"))
    cond = TensorDict({"feature1": f["feature1"], "feature2": f["feature2"]}, batch_size=1)
    out = {}

    def model(fp32, device):
        m = LatentSeqMMFlowModel(**config["denoiser"]["args"])
        if fp32:
            m.float()
            m.dtype = torch.float32
        m.load_state_dict(load_file(str(weights / "denoiser.safetensors")), strict=True)
        turbo._inject(m, turbo.LORA_RANK, turbo.LORA_ALPHA)
        missing, unexpected = m.load_state_dict(lora, strict=False)
        assert not unexpected
        return m.eval().to(device)

    def velocity(m, device, phase, t):
        turbo._set_active(m, phase)
        x = TensorDict({"latent": f["latent_noise"], "camera": f["camera_noise"]}, batch_size=1).to(device)
        v = m(x, torch.tensor([t], device=device, dtype=torch.float32), cond.to(device))
        return v["latent"].float().cpu(), v["camera"].float().cpu()

    with torch.no_grad():
        start = time.time()
        m32 = model(True, "cpu")
        for phase, t in (("slow", 1000.0), ("fast", 400.0)):
            out[f"v32_{phase}_latent"], out[f"v32_{phase}_camera"] = velocity(m32, "cpu", phase, t)
        del m32
        print(f"fp32 turbo forwards (cpu): {time.time() - start:.1f} s")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        m16 = model(False, device)
        for phase, t in (("slow", 1000.0), ("fast", 400.0)):
            out[f"v16_{phase}_latent"], out[f"v16_{phase}_camera"] = velocity(m16, device, phase, t)
        # the 5-step turbo sampler of turbo.py (patch_pipeline_for_turbo installs it as idol.pipeline.sample_flow)
        import idol.pipeline as ip
        original = ip.sample_flow

        class Holder:  # patch_pipeline_for_turbo wants a pipeline with .denoiser; reuse its sampler only
            denoiser = m16
        turbo._inject = lambda *a, **k: None  # already injected
        turbo.load_file = lambda *_: {}
        turbo.patch_pipeline_for_turbo(Holder, REPO)
        noise = TensorDict({"latent": f["latent_noise"], "camera": f["camera_noise"]}, batch_size=1).to(device)
        start = time.time()
        sample = ip.sample_flow(m16, noise, cond.to(device), None, steps=5, rescale_t=3.0, cfg_strength=0.0, verbose=False)
        print(f"5-step turbo sampling ({device}): {time.time() - start:.1f} s")
        ip.sample_flow = original
        out["turbo_latent"], out["turbo_camera"] = sample["latent"].float().cpu(), sample["camera"].float().cpu()
        del m16
        octree = OctreeProbabilityFixedlenDecoder(**config["decoder"]["args"])
        octree.load_state_dict(load_file(str(weights / "decoder.safetensors")), strict=True)
        gs = GaussianFixedlenDecoder(**config["decoder_gs"]["args"])
        gs.load_state_dict(load_file(str(weights / "decoder_gs.safetensors")), strict=True)
        octree, gs = octree.eval().to(device), gs.eval().to(device)
        latent = sample["latent"].float()
        torch.manual_seed(42)
        anchors = Octree.sample(octree, latent, num_points=262144 // 32, level=config["max_voxel_level"])
        gs.to_representation(anchors, gs(anchors, latent))[0].save_ply(str(root / "parity" / "avocado_turbo.ply"))
    path = root / "parity" / "turbo_avocado.safetensors"
    save_deterministic(out, path, meta("turbo_avocado", 0))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
