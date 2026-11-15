# sperid-swift-mlx

Swift / MLX ports of the [speridlabs](https://huggingface.co/speridlabs) models for Apple Silicon, starting with **Idol**: one picture of an object in, up to a million 3D Gaussians out, written as a standard 3DGS `.ply`, with no Python at inference time.

> **Status — 0.2 (2026-11-15): one image → 3D splat, entirely in Swift.** `idol generate --turbo` turns a picture into a 262 144-Gaussian `.ply` in about 25 s on an M3 Max (5 steps, idol-turbo), or ~10 min with the 50-step base model. Every stage is checked against the Python SDK: fp32 anchors bit-exact, DINOv3 1.8e-6, VAE 1.6e-5, 50-step latent 5.9e-3, turbo latent 2.5e-3. **Idol Studio** (`idol-studio`) is the Space on your Mac: same settings and examples, RealityKit 27 splat viewer. Background removal uses Apple Vision for now (BiRefNet port pending).

| Documentation | |
|---|---|
| [Porting plan](PLAN.md) | architecture facts, milestones, pitfalls, budgets (French, one file per section) |
| [Command line](docs/CLI.md) | every `idol` command |
| [Swift API](docs/API.md) | `IdolCore` / `SperidCore` types, the pipeline to come |
| [Weights and licences](docs/Weights.md) | pinned files, SHA-256, the gated encoders |
| [Reference configurations](docs/References.md) | the planned `<bits>bit-fast\|lean` matrix (to be measured) |
| [Benchmarks](docs/Benchmarks.md) | protocol and the expected costs, until measured |
| [Engineering knowledge base](docs/knowledge/index.md) | log, decisions, pitfalls (OKF, French) |

## The models

| Model | Task | Port |
|---|---|---|
| [`speridlabs/idol`](https://huggingface.co/speridlabs/idol) | image → 3D Gaussian splat (1.8 B parameters across 6 networks) | **in progress** — Jalons 1-4 |
| [`speridlabs/idol-turbo`](https://huggingface.co/speridlabs/idol-turbo) | Idol in 5 guidance-free steps (phase LoRAs) | Jalon 5 |
| [`speridlabs/iris-3b`](https://huggingface.co/speridlabs/iris-3b), [`iris-z`](https://huggingface.co/speridlabs/iris-z) | pixel-space text-to-image; joint image + depth | planned ([plan/15](plan/15-famille-sperid.md)) |

## Idol, stage by stage

```
image ─▶ BiRefNet cut-out, square crop, 1024² on black
      ─▶ DINOv3 ViT-H+/16 tokens [4101×1280]  +  FLUX.2 VAE tokens [4101×128]
      ─▶ rectified-flow DiT over 8192 latent tokens + 1 camera token (50 steps, CFG 3; turbo: 5 steps)
      ─▶ octree descent to 256³ (CPU + 8 forwards) ─▶ num_gaussians/32 anchors
      ─▶ 32 Gaussians per anchor ─▶ .ply (y-up) + camera.json
```

Expected costs on an M3 Max are in [plan/03-budget.md](plan/03-budget.md): 150-300 s for the 50-step base model and 8-16 s in turbo, **calculated from the dimensions with an assumed 7-15 TFLOPS, not measured**.

## Getting started

```bash
# build (xcodebuild, never `swift build` for a binary: MLX's metallib would not be found)
Scripts/check-build.sh                  # → .xcodebuild/Build/Products/Release/idol
Scripts/check-tests.sh                  # weight-free tests
.xcodebuild/Build/Products/Release/idol info              # pinned weights, local copies vs contracts
.xcodebuild/Build/Products/Release/idol schedule --steps 5  # the turbo time grid
Scripts/link-references.sh              # reference/: idol-sdk, transformers (DINOv3), sibling ports
.xcodebuild/Build/Products/Release/idol download --turbo   # 3.2 GB; DINOv3 and FLUX.2-dev are gated (accept their terms, HF_TOKEN)
.xcodebuild/Build/Products/Release/idol generate --turbo --image photo.webp --out outputs
Scripts/check-build.sh idol-studio && .xcodebuild/Build/Products/Release/idol-studio   # the app
```

## Repository layout

```
Sources/SperidCore     Hub files, models directory, safetensors headers, weight contracts (shared)
Sources/IdolCore       Configuration, Sampling, Decoding, Output — Models/, Preprocess/, Pipeline/ to come
Sources/IdolCLI        `idol`
Tests/SperidTests      swift-testing suites (weight-free; checkpoint-backed ones gated on SPERID_MODELS_DIR)
plan/  tasks/          the porting plan and its cards (AGENTS.md is the executor's manual)
Scripts/               build, tests, parity, references; Scripts/reference = Python dumps
docs/                  user documentation; docs/knowledge = engineering knowledge base
```

## Licences

Code: MIT ([LICENSE](LICENSE)). Weights are not in this repository: Idol and Idol Turbo are Apache-2.0; DINOv3 ships under Meta's DINOv3 licence (gated); the FLUX.2 VAE comes from `black-forest-labs/FLUX.2-dev` (gated, see its card). Iris-Z is under the FLUX non-commercial licence. Details in [docs/Weights.md](docs/Weights.md).
