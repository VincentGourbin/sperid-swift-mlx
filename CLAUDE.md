# CLAUDE.md
## Project Overview
Swift MLX port of the Speri Labs models, starting with **Idol** (`speridlabs/idol`, Apache-2.0): one image → BiRefNet cut-out → DINOv3 ViT-H+/16 + FLUX.2 VAE tokens → `LatentSeqMMFlowModel` rectified flow (8192 latent tokens + camera) → octree anchors → 32 Gaussians per anchor → standard 3DGS `.ply`. `idol-turbo` (phase LoRAs, 5 guidance-free steps) is Jalon 5; `iris-3b` / `iris-z` are planned only (`plan/15-famille-sperid.md`).
**PLAN.md is the porting spec** (a table of contents into `plan/`) — read `plan/02.x` (architecture facts) and `plan/11-pieges.md` (pitfall checklist) before touching model code. Upstream reference: https://github.com/speridlabs/idol-sdk @ `8db7003` (`reference/idol-sdk`, created by `Scripts/link-references.sh`).
## Build & Run
Build with `xcodebuild`, NOT `swift build` (SPM binaries cannot find MLX's `default.metallib`):
  Scripts/check-build.sh   # Release, binary: .xcodebuild/Build/Products/Release/idol
`swift build` is for compile checks only.
## Tests
`Scripts/check-tests.sh [Suite]` → `Scripts/run-tests.sh` (serial swift-testing: ABBA deadlock in mlx-swift between compiled functions and vjp; wall-clock watchdog; relays every `SPERID_*` variable as `TEST_RUNNER_SPERID_*`). Tiers: weight-free (always green), checkpoint-backed (`Real*Tests.swift`, gated on `SPERID_MODELS_DIR`), end-to-end (gallery images). `xcodebuild test` relaunches crashed workers and still prints ✔ — trust `xcrun xctest`.
## Model weights
`$SPERID_MODELS_DIR/<repo id>/<path>` (default `~/Library/Caches/models`), every file pinned in `IdolWeights` (revision + SHA-256); `idol info` checks local copies against their header contracts. DINOv3 is gated; access was **granted** on 2026-11-15 (official repo, in the HF cache with `FLUX.2-dev/vae`): see `docs/knowledge/decisions/weights-source.md`. This repository is **public** (since 0.3.0): never commit a local absolute path, a weight, a real fixture, a gated header or a personal image; `.local-runs/` (gitignored) holds machine-specific logs.
## Architecture
Sources/SperidCore (Hub files, safetensors headers, weight contracts — shared with future Iris ports); Sources/IdolCore/{Configuration,Sampling,Decoding,Output} now, {Models/{Layers,Denoiser,Decoders,Encoders,BiRefNet},Preprocess,Pipeline} as the milestones land; Sources/IdolCLI (`idol`); Tests/SperidTests; parity/ (tiny fixtures, Python-generated only); Scripts/reference (Python dumps).
## Key implementation facts (do not re-derive)
See `plan/02.2`–`02.8`: RoPE applied BEFORE q/k norm and on interleaved pairs in the denoiser (rotate_half in DINOv3); MultiHeadRMSNorm = L2-normalize × gamma × √64 in fp32; GELU tanh; time embedding cos-then-sin on 1000·t; uncond = zeros before the cond projections; VAE posterior is sampled, patchified 2×2, BN-normalized (eps 1e-4), 5 zero rows prepended; octree runs on the CPU with 8 GPU forwards; every random draw goes through `NoiseSource` (MLX cannot replay torch's stream).
## Performance work — read before measuring
GPU clock state moves results up to 10×: cool down before every point, A/B/B/A, control sample, judge against ±2 % spread. Profiler (`swift-mlx-profiler`) first, `print` never. Before timing: `pgrep` for other GPU jobs, prequantized pack already exported, outputs under `.local-runs/bench.noindex/` (Spotlight indexes fresh files). Expected costs are in `plan/03-budget.md` (CALCUL with an assumed 7-15 TFLOPS until T-2.5 measures it). Record durable findings in docs/knowledge/log.md.
## Engineering Knowledge Base
docs/knowledge/ is an OKF bundle (index.md, log.md, decisions/, pitfalls/, benchmarks/, investigations/), in French. `docs/knowledge/agent-memory.md`: read it at the start of a session.
## Execution by agents
`AGENTS.md` (French) is the executor's manual: one card (`tasks/T-x.y.md`) at a time, one `plan/` file per step, validations in order, ⛔ GATE = stop and ask in `tasks/ASK.md`.
