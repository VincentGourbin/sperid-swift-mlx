# Changelog

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
