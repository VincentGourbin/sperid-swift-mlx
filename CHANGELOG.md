# Changelog

## 0.1.0 — 2026-11-15

- Package `SperidSwift`: `SperidCore` (pinned `HubFile`s, models directory, safetensors header reader, `WeightContract`), `IdolCore`, CLI `idol`, mlx-swift 0.32.3 pinned exact.
- `IdolCore`: `IdolConfig` (the published `config.json`), `IdolRequest` (SDK defaults and bounds, `.turbo`), `IdolWeights` (every file pinned to a Hub revision with its SHA-256, tensor-count and anchor-shape contracts from the published headers), `FlowSchedule` (Euler grid, network time, turbo phase), `NoiseSource` (every random draw injectable), `SystematicSampler` (octree resampling, CPU), `GaussianCloud` (activations), `PLYWriter` (standard 3DGS PLY, z-up → y-up), `CameraPose`.
- CLI: `idol info` (local copies vs contracts, headers only), `idol schedule`.
- Plan rév. 1 (`PLAN.md`, `plan/`), cards for Jalon 1 (`tasks/`), knowledge base (`docs/knowledge/`).
