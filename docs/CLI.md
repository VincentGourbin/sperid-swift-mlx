# `idol` command line

Build with `Scripts/check-build.sh`; the binary is `.xcodebuild/Build/Products/Release/idol`.

## Commands

| Command | Status | What it does |
|---|---|---|
| `idol info [--models-dir DIR]` | available | lists every pinned weight file, whether it is present under the models directory, and checks present safetensors against their contract (tensor count, anchor shapes) by reading the header only |
| `idol schedule [--steps N] [--rescale-t R]` | available | prints the Euler time grid: t, next t, the network time (1000·t), the turbo expert of each step |
| `idol download [--turbo]` | available | downloads the pinned files with resume and SHA-256 checks (`HF_TOKEN` for gated repositories) |
| `idol decode --latent FILE [--key latent] [--num-gaussians N] [--noise FILE --noise-prefix P] [--seed S] [--fp32] --out x.ply` | available | latent [1, 8192, 16] → octree anchors → Gaussians → `.ply`; `--noise` replays recorded draws (parity) |
| `idol denoise --cond FILE` | T-2.7 | image tokens → latent and camera |
| `idol encode --image FILE` | T-3.5 | preprocessed image → DINOv3 and FLUX.2 VAE tokens |
| `idol generate --image FILE [--mode synthetic\|real] [--seed] [--steps] [--cfg] [--num-gaussians] [--turbo] [--reference ID]` | T-4.6, T-5.3, T-6.4 | the whole pipeline; writes `<name>.ply`, `<name>_input.png`, `<name>_camera.json` like the Python SDK |
| `idol export --ply x.ply [--glb y.glb] [--usdz y.usdz] [--resolution 192]` | available | coloured mesh from a splat: the transmittance-0.5 surface seen from above and the sides; GLB with vertex colours, USDZ with a baked texture atlas and a UsdPreviewSurface (passes `usdchecker --arkit`) |
| `idol generate … --glb --usdz` | available | also writes `<name>.glb` / `<name>.usdz` |
| `idol export --ply x.ply --stl y.stl [--height 100] [--thickness 1.2] [--base flat\|plinth\|none] [--max-triangles 200000] [--cells 192] [--smoothing 60] [--print-min-opacity 0]` | available | printable STL: one closed, solid piece in millimetres, z up, on the bed; prints its `PrintReport` (`IDOL_PRINT_TIMING=1` times each stage) |
| `idol generate … --stl` | available | also writes `<name>.stl` with the default print settings |
| `idol references` | T-6.4 | the reference configurations |
| `idol parity decode` | available | Swift against the Python fixtures (`Scripts/check-parity.sh decode` adds the PLY check) |

The models directory is `--models-dir`, else `$SPERID_MODELS_DIR`, else `~/Library/Caches/models`; files live at `<dir>/<repo id>/<path>`.

## Defaults (same as the Python SDK)

| Option | Default | Notes |
|---|---|---|
| `--mode` | `synthetic` | crop margin 1.2 (`real`: 1.7, photos) |
| `--seed` | 42 | Swift and torch draw different numbers from the same seed: parity replays the Python draws |
| `--steps` | 50 | turbo: 5 |
| `--cfg` | 3.0 | turbo: 0 (guidance folded into the LoRAs) |
| `--rescale-t` | 3.0 | |
| `--num-gaussians` | 262 144 | 32 768 to 1 048 576, rounded down to a multiple of 32 |
