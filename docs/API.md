# Swift API

Two libraries: **`IdolCore`** (the pipeline, the engine, the exports — macOS 15+, iOS 27+) and **`IdolUI`**
(RealityKit 27: the splat as an entity, a SwiftUI view, an offscreen renderer).

```swift
.package(url: "https://github.com/VincentGourbin/sperid-swift-mlx", from: "0.3.3")
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
| `engine.generate(_ image: IdolImage, settings:progress:) async throws -> IdolResult` | one generation at a time (calls are serialised); **cancelling the calling task** stops it before the next denoising step: ≈ 0.2 s in turbo, up to half a guided step in `.quality` (one forward, ≈ 5-7 s: the check also sits between the conditional and the negative pass) |
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
| `mesh(options:) -> SplatMesh` | coloured mesh, y-up, ≈ 1 m: the surface a render shows (`MeshExtractionOptions.resolution`, `smoothingIterations`) |
| `writeUSDZ(to:options:)` | textured mesh, `UsdPreviewSurface`, passes `usdchecker --arkit` |
| `writeGLB(to:options:)` | mesh with vertex colours (`COLOR_0`) |
| `printMesh(options:) -> PrintMesh?` | closed, solid, single-piece mesh in millimetres for a 3D printer, with its `PrintReport` |
| `writeSTL(to:options:) -> PrintReport` | binary STL (mm, z up, standing on z = 0); throws when no matter is left |
| `write(to:name:)` | `<name>.ply`, `<name>_input.png`, `<name>_camera.json`, like `python -m idol.sample` |

Other entry points: `PLYReader.read(_:)` (any 3DGS PLY → `GaussianCloud`), `MeshExtraction.extract(_:options:)`,
`PrintPreparation.prepare(_:options:)`, `GLBWriter`, `USDZWriter`, `STLWriter`.

### 3D printing — `PrintOptions`, `PrintReport`

```swift
let report = try result.writeSTL(to: url, options: PrintOptions(heightMM: 120, minThicknessMM: 1.2, base: .plinth))
guard report.isPrintable else { … }   // watertight, manifold, one shell, positive volume
```

The splat becomes a solid on a voxel grid, then a surface:

| Step | Why |
|---|---|
| every Gaussian as an exact extinction kernel (3σ ellipsoid; a ray through a face-on Gaussian loses its α, as in a 3DGS render) | a blurred, point-sampled density put the surface in the Gaussians' tails: every object came out puffy |
| solid = what a viewer **above or level** cannot see (transmittance 0.5, 17 directions) | the surface sits where a render shows it; the underside, never seen by the model and often open, does not empty the inside; open tops (a foosball's pitch, a cup) stay open |
| keep the largest piece | floaters would print as loose crumbs |
| closing + growth of thin parts to `minThicknessMM` | ears, rods, sheets thinner than a nozzle |
| `base`: `.flat` (cut 2 % of the height), `.plinth` (convex hull of the feet + `plinthMarginMM`), `.none` | something to stand on the bed |
| ambiguity repair (no 2×2×2 block whose matter or void touches only along an edge or a corner) | the surface is then a 2-manifold by construction |
| surface nets, 60 Taubin passes, quadric decimation (link condition, no flip) to `maxTriangles` | no voxel steps, no shrinkage, a file a slicer opens quickly |

`gridCells` (default 192 over the height) sets the finest detail. `PrintReport` gives the size in mm, the volume, the shells, the genus, the
fragments removed and the two checks (each edge shared by two opposite triangles; a single fan per vertex). It
takes 3-8 s on an M3 Max. Colour (3MF) is not written yet.

The USDZ / GLB mesh (`MeshExtraction`) uses the same field and solid (no thickness, no base), keeps pieces
above 1 % of the largest, and colours the vertices from the Gaussians that make the surface.

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
