# Swift API

Two libraries: **`IdolCore`** (the pipeline, the engine, the exports — macOS 15+, iOS 27+) and **`IdolUI`**
(RealityKit 27: the splat as an entity, a SwiftUI view, an offscreen renderer).

```swift
.package(url: "https://github.com/VincentGourbin/sperid-swift-mlx", from: "0.3.2")
```

## The shortest path

```swift
import IdolCore

let engine = try await IdolEngine.load()                                   // load once
let result = try await engine.generate(.url(photo), settings: .turbo)      // ≈ 20 s on an M3 Max
try result.writeUSDZ(to: out)
```

## Weights — `IdolModels`

| | |
|---|---|
| `IdolModels.status(modelsDirectory:includeTurbo:) -> Status` | what is missing (`missing`, `missingBytes`, `isComplete`); encoders already in the Hugging Face cache count as present |
| `IdolModels.download(modelsDirectory:includeTurbo:token:progress:) async throws` | resumable, SHA-256 checked; the token defaults to `HF_TOKEN`, else the file written by `hf auth login` |
| `IdolModels.files(includeTurbo:) -> [HubFile]` | every pinned file (repo, revision, size, SHA-256, gated) |

The default models directory is `$SPERID_MODELS_DIR`, else `~/Library/Caches/models` (`ModelsDirectory.resolve()`).
A gated file the account cannot read throws `SperidError.gatedRepository` with the page where the terms are accepted.

## Generation — `IdolEngine`

| | |
|---|---|
| `IdolEngine.load(modelsDirectory:configuration:) async throws` | loads the six networks (≈ 4 GB at the reference precision); each engine owns one MLX thread (`MLXThread`) |
| `engine.generate(_ image: IdolImage, settings:progress:) async throws -> IdolResult` | one generation at a time (calls are serialised); **cancelling the calling task** stops it before the next denoising step: ≈ 0.2 s in turbo, up to one guided step (≈ 10 s, Fluxforge measured 6.8 s) in `.quality` |
| `engine.hasTurbo` | the idol-turbo LoRAs were found |
| `IdolEngine.Configuration` | `backgroundRemoval: .birefnet` (default, the reference) or `.vision` (Apple Vision, no weights, misses subjects in busy scenes); `memory: .bounded()` (2 GB MLX cache: ≈ 11 GB process peak instead of 34) or `.unbounded` |

`IdolImage`: `.url(URL)`, `.cgImage(CGImage)`, `.rgba(RGBAImage)`. An image with real alpha skips the cut-out.

`IdolProgress` (from the engine's thread): `stage` (`.preprocessing`, `.encoding`, `.denoising`, `.decoding`,
`.done`) and `fraction` (steps done during `.denoising`).

### `IdolSettings`

| Field | Default | Space name |
|---|---|---|
| `mode` | `.synthetic` (renders, generated images) — `.real` for photos | Input type |
| `seed` | `42` — `nil` draws one (returned in `result.seed`) | Seed / Random seed |
| `turbo` | `true` (5 guidance-free steps) | — |
| `steps` | `50` (base model) | Steps |
| `guidance` | `3.0` (base model) | Guidance scale |
| `timeShift` | `3.0` | Time shift |
| `gaussians` | `262_144` (32 768 … 1 048 576) | Gaussians |

Presets: `.turbo` (≈ 20 s), `.quality` (50 steps, guidance 3, ≈ 8 min on an M3 Max). Swift's random stream is not
torch's: the same seed gives a different (equally valid) object than the Python SDK.

### `IdolResult`

| | |
|---|---|
| `cloud: GaussianCloud` | activated Gaussians (xyz, SH DC colour, scale, unit quaternion, opacity), object z-up in [-0.5, 0.5]³ |
| `preprocessed: RGBAImage`, `preprocessedCGImage` | the 1024² image the model saw |
| `camera: CameraPose` | yaw, pitch (radians), radius, fov (degrees) of the input view |
| `seed`, `settings`, `timings` | what ran, seconds per stage |
| `plyData()`, `writePLY(to:)` | standard 3DGS binary PLY (y-down, the COLMAP convention every 3DGS viewer reads) |
| `mesh(options:) -> SplatMesh` | coloured mesh, y-up, ≈ 1 m (density grid + surface nets; `MeshExtractionOptions.resolution`, `isoFraction`) |
| `writeUSDZ(to:options:)` | textured mesh, `UsdPreviewSurface`, passes `usdchecker --arkit` |
| `writeGLB(to:options:)` | mesh with vertex colours (`COLOR_0`) |
| `write(to:name:)` | `<name>.ply`, `<name>_input.png`, `<name>_camera.json`, like `python -m idol.sample` |

Other entry points: `PLYReader.read(_:)` (any 3DGS PLY → `GaussianCloud`), `MeshExtraction.extract(_:options:)`,
`GLBWriter`, `USDZWriter`.

## `IdolUI` (macOS 27, iOS 27, visionOS 27)

```swift
import IdolUI

SplatView(cloud: result.cloud, id: generation)              // RealityView + orbit controls
let entity = try SplatEntity.make(result.cloud)             // add to your own RealityKit scene (upright, y-up)
try await SplatRenderer.render(result.cloud, to: png, inputCamera: result.camera)   // 4 views, offscreen
```

## Lower level

`IdolPipeline` (synchronous, one thread: the CLI uses it), `IdolDenoiser`, `IdolDecoder`, `DINOv3`, `FluxVAEEncoder`,
`BiRefNet`, `FlowSampler`, `NoiseSource` (`SeededNoise`, `ReplayNoise` for parity) are public for research and
parity work. Rule if you use them in an app: every MLX call on **one** thread (`MLXThread`), and nothing lazy
crosses it — evaluate first (pattern MLX-004).
