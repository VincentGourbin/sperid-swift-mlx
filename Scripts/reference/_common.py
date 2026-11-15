"""Shared helpers for the fixture generators (card T-1.3): provenance metadata, deterministic files,
and the noise recorder that turns every random draw of the octree descent into a stored input."""
import datetime
import json
import struct
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
SDK = ROOT / "reference" / "idol-sdk"


def upstream_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=SDK, capture_output=True, text=True,
                          check=True).stdout.strip()


def meta(kind, seed):
    return {"kind": kind, "seed": str(seed), "upstream_sha": upstream_sha(),
            "generated": datetime.date.today().isoformat()}


def save_deterministic(tensors, path, metadata):
    """save_file, then canonical (sorted) header JSON: the Rust writer's key order is randomized
    per process, so two identical runs would otherwise differ byte for byte (YuE2, T-1.3)."""
    tensors = {k: v.detach().contiguous() for k, v in tensors.items()}
    save_file(tensors, str(path), metadata=metadata)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(canonical)))
        f.write(canonical)
        f.write(data)


def randomize(module, generator, scale=0.2):
    """Every parameter and float buffer ~ N(0, scale²): zero-initialised tensors (shift tables,
    offsets) must be exercised too."""
    with torch.no_grad():
        for t in list(module.parameters()) + [b for b in module.buffers() if b.is_floating_point()]:
            t.copy_(torch.randn(t.shape, generator=generator) * scale)


class OctreeNoiseRecorder:
    """Runs the upstream OctreeProbabilityFixedlenDecoder.sample unchanged while controlling its draws.

    - systematic_sample draws one u0 per row through torch.rand(r, 1), grouped by count value in
      `counts.unique(sorted=False)` order. The wrapper recomputes those groups (same call, same
      input) and serves torch.rand from a per-cell table drawn here in cell order, so Swift can
      replay `offset[cell]` without knowing torch's grouping.
    - the final in-cell jitter (torch.rand_like) is drawn from the generator and recorded.
    - the probabilities of each level are recorded, so the descent can be tested apart from the net.
    """

    def __init__(self, decoder_module, generator):
        self.mod = decoder_module
        self.gen = generator
        self.offsets, self.probs, self.jitter = [], [], None

    def run(self, model, cond, num_points, level):
        original_ss, original_rand, original_rand_like = self.mod.systematic_sample, torch.rand, torch.rand_like
        recorder = self

        def wrapped_ss(probs, counts):
            recorder.probs.append(probs.detach().float().reshape(-1, probs.shape[-1]).clone())
            flat = counts.reshape(-1)
            table = torch.rand(flat.numel(), generator=recorder.gen)
            recorder.offsets.append(table.clone())
            unique_n, inv = flat.unique(sorted=False, return_inverse=True)
            groups = [(inv == i).nonzero(as_tuple=False).squeeze(1) for i, n in enumerate(unique_n.tolist()) if n != 0]
            calls = iter(groups)

            def fake_rand(*shape, **kwargs):
                rows = next(calls)
                return table[rows].reshape(-1, 1).to(kwargs.get("device", "cpu"))

            torch.rand = fake_rand
            try:
                return original_ss(probs, counts)
            finally:
                torch.rand = original_rand

        def fake_rand_like(t, dtype=None, **kwargs):
            out = torch.rand(t.shape, generator=recorder.gen, dtype=dtype or t.dtype)
            recorder.jitter = out.clone()
            return out

        self.mod.systematic_sample = wrapped_ss
        torch.rand_like = fake_rand_like
        try:
            return self.mod.OctreeProbabilityFixedlenDecoder.sample(model, cond, num_points=num_points, level=level)
        finally:
            self.mod.systematic_sample = original_ss
            torch.rand_like = original_rand_like
