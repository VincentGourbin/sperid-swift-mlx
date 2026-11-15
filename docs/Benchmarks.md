# Benchmarks: method, expected costs, reproduction

No measurement yet. Raw rows go to [`BENCHMARKS.md`](../BENCHMARKS.md); this page explains them.

## 1. What is measured
Per stage: preprocessing, encoders, denoising (time per step × steps), octree, Gaussian decoder, PLY writing; peak MLX memory and `phys_footprint`; quality as PSNR/SSIM of control renders against the 16-bit reference (card T-6.1).

## 2. Protocol (`CLAUDE.md`, "Performance work")
Release build; `swift-mlx-profiler` only; GPU cooled down before every point; A/B/B/A with a control sample; judge against the ±2 % spread; no other GPU job (`pgrep`); outputs under `.local-runs/bench.noindex/`; the first run after a pack export is discarded.

## 3. Expected costs (calculated from the dimensions, not measured)
| Stage (M3 Max) | Expected |
|---|---:|
| One denoiser forward, 12 294 tokens | ≈ 1.5 s |
| Base model, 50 steps with CFG (100 forwards) | ≈ 150 s |
| Turbo, 5 steps | ≈ 7-8 s |
| DINOv3 ViT-H+ | ≈ 0.7 s |
| Gaussian decoder, 262 k / 1 M Gaussians | ≈ 1 s / ≈ 8 s |

For scale, the model card reports 6.8 s on an H100 and 19.1 s on an RTX 4090 (50 steps, 262 k Gaussians). Derivation in [plan/03-budget.md](../plan/03-budget.md).
