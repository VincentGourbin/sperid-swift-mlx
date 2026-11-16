# Changelog

## Unreleased

- **Sharper meshes (USDZ, GLB) — not puffy any more.** The mesh was the iso-surface of a point-sampled, blurred density at 4 % of its peak: it sat in the Gaussians' tails, 2-3 cells outside the object. Now every Gaussian is an exact extinction kernel and the surface is where a ray loses half its light, as in a render; the solid is what a viewer above or level with the object cannot see, so open tops stay open (the foosball's rods and players are back) and the never-seen underside does not empty the inside. 60 Taubin passes remove the voxel steps. The USDZ also halves in size (29 → 13 MB on the foosball). `MeshExtractionOptions.isoFraction` is ignored (deprecated); `minimumOpacity` defaults to 0.
- **Printable STL** (`IdolResult.writeSTL`, `PrintPreparation`, `idol export --stl`, `idol generate --stl`, Idol Studio's Export menu): the same solid, then minimum thickness, flat base or convex-hull plinth, ambiguity repair, surface nets, smoothing, quadric decimation. Watertight and manifold by construction and checked (`PrintReport`). Fox, foosball, avocado, backpack: one shell each, watertight, manifold, 3-8 s. No Blender, no new dependency.

## 0.3.3 — 2026-11-16

- **Cancellation in `.quality` waits half a step instead of a whole one.** The guided step evaluates its conditional forward on its own and checks cancellation before the negative forward. Bit-identical: same PLY bytes for a 6-step guided generation, same 50-step parity against PyTorch (final latent relL2 0.0030875097, unchanged). Cost: one graph build (13.7 ms) per step; interleaved in one process, 12 steps each, median 11.96 s before vs 11.99 s after, within noise. The engine test bounds the latency by 0.6 × the measured step (5.1 s for a 10.1 s step).

## 0.3.2 — 2026-11-16

- **`IdolEngine` gives its thread back.** `MLXThread`'s loop never returned, so every dropped engine left one parked thread (16 MB of stack reserved) — one per release/reload cycle in an app that frees the engine between objects (reported by Fluxforge Studio). New `MLXThread.stop()` (drains the queued jobs, then the thread exits; `run` throws `CancellationError` and `async` returns `false` afterwards) and `isFinished`; `IdolEngine.deinit` calls `stop()`. Checked on the real weights: 3 load/release cycles leaked 3 threads before (14 → 17), none after (10 → 10).
- **Cancellation latency, documented correctly**: the check sits between denoising steps, so the bound is one step — ≈ 0.2 s in turbo, one guided step (≈ 10-15 s, two full forwards) in `.quality`. The 0.7 s quoted for 0.3.0 was a cancel that landed during the encoders. The engine test now cancels mid-denoising and bounds the latency by the measured step.

## 0.3.1 — 2026-11-15

- **mlx-swift `0.31.6..<0.33.0`** instead of `exact: "0.32.3"`, so that apps pinning 0.31.6 (Fluxforge Studio with flux-2-swift-mlx and yue2-mlx-swift) can resolve the package. Validated at 0.31.6 on the real weights: 62 tests (parity of every network, turbo, `IdolEngine` with cancellation), `idol parity decode|denoise` (fp32 anchors bit-exact, 50-step latent 3.1e-3), an end-to-end generation with USDZ (`usdchecker --arkit` passes). The upper end (0.32.3) is the v0.3.0 validation. No code change.

## 0.3.0 — 2026-11-15 — public release

- **Whole pipeline in Swift/MLX**, checked against the Python SDK on real weights: BiRefNet (Swin-L, deformable convolution), DINOv3 ViT-H+/16, FLUX.2 VAE encoder, denoiser (50 steps CFG or idol-turbo 5 steps), octree + Gaussian decoders.
- **API**: `IdolEngine` (async, cancellable, own MLX thread), `IdolModels` (status, resumable verified downloads, gated repos via `hf auth login`), `IdolSettings` (`.turbo`, `.quality`), `IdolResult` (PLY, USDZ, GLB).
- **IdolUI**: `SplatEntity`, `SplatView` (RealityKit 27 Gaussian splats), `SplatRenderer` (offscreen).
- **Exports**: USDZ (texture atlas, UsdPreviewSurface, passes `usdchecker --arkit`) and GLB (vertex colours) from any splat.
- **Idol Studio**: the Space on the Mac (settings, examples, viewer, export).
- Performance: fused rotary + RMSNorm (−4 %, bit-exact), bounded MLX cache (process peak 34 → 11 GB).

## 0.1.0 — 2026-11-15

- Package `SperidSwift`: `SperidCore` (pinned `HubFile`s, models directory, safetensors header reader, `WeightContract`), `IdolCore`, CLI `idol`, mlx-swift 0.32.3 pinned exact.
- `IdolCore`: `IdolConfig` (the published `config.json`), `IdolRequest` (SDK defaults and bounds, `.turbo`), `IdolWeights` (every file pinned to a Hub revision with its SHA-256, tensor-count and anchor-shape contracts from the published headers), `FlowSchedule` (Euler grid, network time, turbo phase), `NoiseSource` (every random draw injectable), `SystematicSampler` (octree resampling, CPU), `GaussianCloud` (activations), `PLYWriter` (standard 3DGS PLY, z-up → y-up), `CameraPose`.
- CLI: `idol info` (local copies vs contracts, headers only), `idol schedule`.
- Plan rév. 1 (`PLAN.md`, `plan/`), cards for Jalon 1 (`tasks/`), knowledge base (`docs/knowledge/`).

## Unreleased — Jalon 1 (2026-11-15)

- `HubDownloader` (resume, SHA-256, gated-repository error) and `idol download [--turbo]`.
- Decoder layers (`LayerNorm32`, `MultiHeadRMSNorm`, sinusoidal and point embedders, attention, both cross blocks), `OctreeDecoder` + CPU `OctreeDescent`, `GaussianDecoder` + `GaussianAssembly`, `IdolDecoder`; `ReplayNoise` / `SeededNoise`.
- `idol decode`, `idol parity decode`; tiny fixtures from the upstream classes, real fixtures without the gated encoders (unconditional latent from the real denoiser).
- Parity: fp32 anchors bit-exact, Gaussian decoder 5e-6 (fp32), PLY 1.9e-6 from `save_ply`.
