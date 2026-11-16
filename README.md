# sperid-swift-mlx

Swift / MLX port of **[Idol](https://huggingface.co/speridlabs/idol)** by [Speri Labs](https://huggingface.co/speridlabs) for Apple Silicon: **one picture of an object in, a 3D Gaussian splat out** — entirely on the Mac, no Python at run time. Exports the standard 3DGS `.ply`, a textured **USDZ** (Quick Look, AR), a **GLB** (Blender, web) and a printable **STL** (one closed, solid piece in millimetres).

> Independent community port, not affiliated with Speri Labs. The weights are theirs (and Meta's and Black Forest Labs' for the two encoders); they are downloaded under your own Hugging Face account and their licences apply — see [Licences](#licences).

| Turbo, 5 steps, ≈ 20 s on an M3 Max | |
|---|---|
| A photo of a living room → the foosball table only (BiRefNet cut-out), then 262 144 Gaussians, then a USDZ mesh | `idol generate --turbo --mode real --image photo.jpg --usdz` |

## What you get

- **`IdolCore`** — the whole pipeline in Swift/MLX: BiRefNet cut-out, DINOv3 ViT-H+/16 and FLUX.2 VAE encoders, the rectified-flow denoiser (50-step base model or **idol-turbo**, 5 steps), octree + Gaussian decoders, PLY / USDZ / GLB writers, and a small async API (`IdolEngine`).
- **`IdolUI`** — the splat as a RealityKit 27 entity (`GaussianSplatComponent`), a SwiftUI `SplatView` with orbit controls, an offscreen renderer for thumbnails.
- **`idol`** — the command line; **`idol-studio`** — a macOS app that mirrors the [Idol Space](https://huggingface.co/spaces/speridlabs/idol): same settings, same 94 examples, live 3D viewer, export menu.

Every stage is checked against the Python SDK on real weights and real images:

| Stage | Swift vs PyTorch |
|---|---|
| BiRefNet alpha (fp32) | 1.0e-6 (uint8 alpha within 1 level) |
| DINOv3 tokens (fp32) | 1.8e-6 |
| FLUX.2 VAE tokens (fp32) | 1.6e-5 |
| Denoiser, one forward (fp32) | 7.7e-5 |
| Final latent, 50 steps CFG 3 (fp16) | 5.9e-3 — same camera |
| Final latent, turbo 5 steps (fp16) | 2.5e-3 |
| Octree anchors (fp32) | bit-identical (8 192 anchors) |
| Gaussian decoder (fp32) | 5.0e-6 |

Measured on an M3 Max 96 GB (in session, see [benchmarks](docs/knowledge/benchmarks/m3max-profile-2026-11-15.md)): turbo ≈ 22 s per object, base model ≈ 8 min, process peak ≈ 11 GB with the default memory policy.

## Use it in an app

```swift
// Package.swift
.package(url: "https://github.com/VincentGourbin/sperid-swift-mlx", from: "0.3.3"),
// target dependencies: .product(name: "IdolCore", package: "sperid-swift-mlx"),
//                      .product(name: "IdolUI", package: "sperid-swift-mlx")   // RealityKit view (macOS / iOS 27)
```

```swift
import IdolCore

// 1. weights (≈ 7.5 GB, once). DINOv3 and FLUX.2-dev are gated: accept their terms on huggingface.co,
//    then `hf auth login` (or HF_TOKEN) — the token is picked up automatically.
if !IdolModels.status().isComplete { try await IdolModels.download { print($0.file, $0.writtenBytes) } }

// 2. load once (≈ 4 GB in memory), generate many; cancelling the Task stops the generation
let engine = try await IdolEngine.load()
let result = try await engine.generate(.url(photoURL), settings: IdolSettings(mode: .real)) { progress in
    print(progress.stage, progress.fraction)
}

// 3. outputs
try result.writePLY(to: folder.appending(path: "object.ply"))     // 3DGS viewers, Blender 5.3
try result.writeUSDZ(to: folder.appending(path: "object.usdz"))   // Quick Look / AR (passes usdchecker --arkit)
try result.writeGLB(to: folder.appending(path: "object.glb"))     // Blender, three.js, Unity
try result.writeSTL(to: folder.appending(path: "object.stl"))     // 3D printer: watertight, manifold, 100 mm high
```

```swift
import IdolUI
SplatView(cloud: result.cloud, id: generation)        // SwiftUI, RealityKit 27 Gaussian splats, orbit controls
```

`IdolSettings.turbo` (default: 5 guidance-free steps) and `IdolSettings.quality` (50 steps, guidance 3) are the two presets; every Space setting is there (`mode`, `seed`, `steps`, `guidance`, `timeShift`, `gaussians`). Full reference: [docs/API.md](docs/API.md).

## Command line and app

```bash
Scripts/check-build.sh                       # Release build (xcodebuild: MLX needs its metallib)
B=.xcodebuild/Build/Products/Release
$B/idol download                             # weights; gated encoders need `hf auth login`
$B/idol generate --turbo --image photo.jpg --mode real --out outputs --usdz --glb
$B/idol export --ply outputs/photo.ply --usdz photo.usdz      # mesh from any 3DGS .ply
Scripts/check-build.sh idol-studio && $B/idol-studio          # the app
```

Requirements: Apple Silicon, macOS 15+ for the library and the CLI (macOS 27 for the RealityKit splat viewer), Xcode 27. Memory: ≈ 11 GB process peak in turbo with the default policy (measured on 96 GB; a 16 GB Mac is untested).

## How it works

```
image ─▶ BiRefNet cut-out, square crop, 1024² on black
      ─▶ DINOv3 ViT-H+/16 tokens [4101 × 1280]  +  FLUX.2 VAE tokens [4101 × 128]
      ─▶ rectified-flow DiT over 8 192 latent tokens + 1 camera token (50 steps CFG 3, or turbo: 5 steps)
      ─▶ octree descent to 256³ (CPU + 8 forwards) ─▶ num_gaussians / 32 anchors
      ─▶ 32 Gaussians per anchor ─▶ .ply  ─▶ (optional) density grid + surface nets ─▶ .usdz / .glb
```

The port's engineering record — plan, decisions, pitfalls (Swift's `Float.pi` rounded toward zero, the y-down 3DGS PLY, MLX graphs crossing threads…) — lives in [`plan/`](PLAN.md), [`tasks/`](tasks/INDEX.md) and [`docs/knowledge/`](docs/knowledge/index.md) (in French).

| Documentation | |
|---|---|
| [Swift API](docs/API.md) | `IdolEngine`, `IdolModels`, `IdolSettings`, `IdolResult`, `IdolUI` |
| [Command line](docs/CLI.md) | every `idol` command |
| [Weights and licences](docs/Weights.md) | pinned files, SHA-256, the gated encoders |
| [Benchmarks](docs/Benchmarks.md) | protocol and measurements |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | code the port follows, weight licences |

## Licences

- **Code**: MIT ([LICENSE](LICENSE)). It follows Apache-2.0 and MIT reference implementations (idol-sdk, transformers, diffusers, BiRefNet, Swin): [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Weights**, never in this repository: Idol and Idol Turbo — Apache-2.0 (Speri Labs); DINOv3 ViT-H+/16 — Meta's DINOv3 licence (gated); FLUX.2 VAE — taken from `black-forest-labs/FLUX.2-dev`, FLUX Non-Commercial License (gated). Read those terms before any commercial use.
