#!/usr/bin/env python3
"""Tiny fixtures (card T-1.3): the upstream idol classes at small sizes, random weights, fp32.

Run from the repo root:  .venv-ref/bin/python -I Scripts/reference/tiny_fixtures.py [out_dir]
Writes <out_dir>/tiny_layers.safetensors (default parity/). Keys: `<case>.<role>.<name>` with role
`w` (module state_dict, upstream names), `in` (inputs) or `out` (outputs).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OctreeNoiseRecorder, ROOT, meta, randomize, save_deterministic  # noqa: E402

import idol.models.decoder as decoder_mod  # noqa: E402
from idol.models import layers as L  # noqa: E402

SEED = 1234
C, H, CTX = 64, 4, 16          # channels, heads (head_dim 16), latent channels
REP = {"num_gaussians": 32, "lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0, "_scaling": 1.0,
                                     "_rotation": 0.1},
       "perturbe_size": 1.5, "3d_filter_kernel_size": 0.0009, "scaling_bias": 0.004, "opacity_bias": 0.1}


def fp32(module):
    """The decoders cast their blocks to fp16 (`self.dtype`); tiny fixtures stay fp32 end to end."""
    module.float()
    if hasattr(module, "dtype"):
        module.dtype = torch.float32
    return module.eval()


def add(out, case, module=None, inputs=None, outputs=None):
    if module is not None:
        for k, v in module.state_dict().items():
            if v.is_floating_point():
                out[f"{case}.w.{k}"] = v.float()
    for role, d in (("in", inputs or {}), ("out", outputs or {})):
        for k, v in d.items():
            out[f"{case}.{role}.{k}"] = v.float() if v.is_floating_point() else v


@torch.no_grad()
def build():
    g = torch.Generator().manual_seed(SEED)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    out = {}

    x = rn(2, 7, C)
    for name, affine, eps in (("ln_affine", True, 1e-5), ("ln_plain", False, 1e-6)):
        m = fp32(L.LayerNorm32(C, elementwise_affine=affine, eps=eps))
        if affine:
            randomize(m, g)
        add(out, name, m, {"x": x}, {"y": m(x)})

    m = fp32(L.MultiHeadRMSNorm(16, H)); randomize(m, g)
    xh = rn(2, 7, H, 16)
    add(out, "rmsnorm", m, {"x": xh}, {"y": m(xh)})

    m = fp32(L.FeedForwardNet(C)); randomize(m, g)
    add(out, "ffn", m, {"x": x}, {"y": m(x)})

    pts = torch.rand(50, 3, generator=g)
    m = fp32(L.PcdAbsolutePositionEmbedderv2(C, in_channels=3))
    add(out, "pos_v2", None, {"x": pts}, {"y": m(pts)})

    t = torch.tensor([0.0, 3.5, 256.0, 1000.0])
    for name, kw in (("time_embed", {}), ("level_embed", {"max_period": 1024, "two_pi": True})):
        m = fp32(L.SinusoidalEmbedder(C, **kw)); randomize(m, g)
        add(out, name, m, {"t": t}, {"y": m(t)})

    ctx = rn(2, 11, CTX)
    m = fp32(L.MultiHeadAttention(C, H)); randomize(m, g)
    add(out, "self_attn", m, {"x": x}, {"y": m(x)})
    m = fp32(L.MultiHeadAttention(C, H, ctx_channels=CTX, cross=True)); randomize(m, g)
    add(out, "cross_attn", m, {"x": x, "context": ctx}, {"y": m(x, ctx)})

    mod = rn(2, 6 * C)
    m = fp32(L.ModulatedTransformerCrossOnlyBlock(C, ctx_channels=CTX, num_heads=H)); randomize(m, g)
    add(out, "cross_only_block", m, {"x": x, "mod": mod, "context": ctx}, {"y": m(x, mod, ctx)})
    m = fp32(L.TransformerCrossBlock(C, ctx_channels=CTX, num_heads=H)); randomize(m, g)
    add(out, "cross_block", m, {"x": x, "context": ctx}, {"y": m(x, ctx)})

    # Octree decoder: one forward, then the full descent with recorded draws.
    latent = rn(1, 32, CTX)
    m = fp32(decoder_mod.OctreeProbabilityFixedlenDecoder(C, CTX, num_blocks=2, num_heads=H)); randomize(m, g, 0.1)
    cells = torch.rand(1, 9, 3, generator=g)
    level = torch.tensor([8])
    add(out, "octree", m, {"x": cells, "level": level.float(), "latent": latent},
        {"logits": m(cells, level, latent)})
    rec = OctreeNoiseRecorder(decoder_mod, g)
    anchors = rec.run(m, latent, num_points=256, level=4)
    descent = {"anchors": anchors, "jitter": rec.jitter}
    for i, (o, p) in enumerate(zip(rec.offsets, rec.probs)):
        descent[f"offsets_{i + 1}"], descent[f"probs_{i + 1}"] = o, p
    add(out, "descent", None, {}, descent)

    # Gaussian decoder: forward + to_representation (activated Gaussian of the first object).
    m = fp32(decoder_mod.GaussianFixedlenDecoder(3, C, CTX, num_blocks=2, num_heads=H, representation_config=REP))
    randomize(m, g, 0.1)
    pts = torch.rand(1, 12, 3, generator=g)
    feats = m(pts, latent)
    gs = m.to_representation(pts, feats)[0]
    add(out, "gaussian", m, {"points": pts, "latent": latent},
        {"features": feats, "xyz": gs.xyz, "features_dc": gs.features_dc.reshape(-1, 3), "scaling": gs.scaling,
         "rotation": gs.rotation, "opacity": gs.opacity})
    return out


def main():
    torch.set_num_threads(1)
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = build()
    path = out_dir / "tiny_layers.safetensors"
    save_deterministic(tensors, path, meta("tiny_layers", SEED))
    print(f"wrote {path} ({len(tensors)} tensors, {path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
