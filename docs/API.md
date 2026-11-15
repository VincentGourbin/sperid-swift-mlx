# `IdolCore` / `SperidCore` Swift API

The public surface grows with the milestones; this page lists what exists, then what is planned.

## Available (0.1.0)

| Type | Module | Role |
|---|---|---|
| `IdolConfig` | IdolCore | the published `config.json`; `conditionTokenCount` (4101), `jointTokenCount` (12 294) |
| `IdolRequest`, `IdolCropMode` | IdolCore | the SDK's parameters and defaults; `.turbo`; `validate()`; `anchorCount()` |
| `IdolWeights` | IdolCore | every file pinned (`HubFile`: repo, revision, size, SHA-256) and its `WeightContract` |
| `FlowSchedule`, `TurboPhase` | IdolCore | time grid `t ← r·t / (1 + (r − 1)·t)`, `networkTime(i)`, `turboPhase(i)` |
| `NoiseSource` | IdolCore | protocol behind every random draw (latent, camera, VAE posterior, octree offsets, cell jitter) |
| `SystematicSampler` | IdolCore | `split(probabilities:count:offset:)` — the octree's systematic resampling |
| `GaussianCloud` | IdolCore | activated Gaussians; `activate(raw…)` is the CPU reference of the decoder activations |
| `PLYWriter` | IdolCore | `write(_:to:)` — standard 3DGS PLY, y-up, inverse activations with the SDK's clamps |
| `CameraPose` | IdolCore | `init(latent:)` — yaw, pitch, radius, fov from the 5-channel camera token |
| `HubFile`, `ModelsDirectory`, `SafetensorsHeader`, `WeightContract`, `SperidError` | SperidCore | shared plumbing |

```swift
import IdolCore

let request = IdolRequest(mode: .real, steps: 50)
try request.validate()
let schedule = FlowSchedule(steps: request.steps, rescaleT: request.rescaleT)
// … the pipeline (Jalon 4) produces a GaussianCloud:
try PLYWriter.write(cloud, to: URL(fileURLWithPath: "object.ply"))
```

## Planned

```swift
let pipeline = try await IdolPipeline.load(modelsDirectory: dir, profile: .named("16bit-fast")!)
let output = try await pipeline.generate(image: cgImage, request: .init(mode: .real)) { progress in … }
try output.write(to: folder, name: "photo")   // photo.ply, photo_input.png, photo_camera.json
```

- `BackgroundRemover` protocol (BiRefNet, Apple Vision, or the app's own mask); skipped when the image has real alpha.
- Cancellation checked before every denoiser evaluation and between octree levels; one synchronisation per step.
- Stage-scoped residency in `lean` profiles (encoders released before denoising).
