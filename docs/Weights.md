# Weights and licences

Nothing is committed: files are downloaded into `$SPERID_MODELS_DIR/<repo id>/<path>` (default `~/Library/Caches/models`) and checked against the revision and SHA-256 pinned in `IdolWeights` (Hub tree API, 2026-11-15).

| File | Size | Precision stored | Licence | Gated |
|---|---:|---|---|---|
| `speridlabs/idol` `denoiser.safetensors` | 1.48 GB | F32 | Apache-2.0 | no |
| `speridlabs/idol` `decoder.safetensors` (octree) | 0.20 GB | F32 | Apache-2.0 | no |
| `speridlabs/idol` `decoder_gs.safetensors` | 0.95 GB | F32 | Apache-2.0 | no |
| `speridlabs/idol` `birefnet.safetensors` | 0.44 GB | F16 | Apache-2.0 | no |
| `speridlabs/idol-turbo` `idol_turbo_lora.safetensors` (v5) | 0.12 GB | F32 | Apache-2.0 | no |
| `facebook/dinov3-vith16plus-pretrain-lvd1689m` | ≈ 1.7 GB | — | DINOv3 licence (Meta) | **yes** |
| `black-forest-labs/FLUX.2-dev` `vae/` (encoder used) | 0.34 GB | F32 | see the FLUX.2-dev card | **yes** |
| `Comfy-Org/Idol` `clip_vision/dino_v3_vit_h_bf16.safetensors` | 1.68 GB | BF16 | labelled Apache-2.0 by the repack | no |
| `Comfy-Org/Idol` `vae/flux2-vae.safetensors` | 0.34 GB | F32 | labelled Apache-2.0 by the repack | no |

The Comfy repack is an ungated copy of the two encoders; whether it is used is an open decision ([decisions/weights-source](knowledge/decisions/weights-source.md)). `idol info` reports, for each file, `absent`, `present`, `ok` (contract verified) or `MISMATCH`.

Contracts (tensor counts read from the published headers): denoiser 560, octree decoder 60, Gaussian decoder 328, BiRefNet 742, turbo LoRA 448, DINOv3 615. Rerun `python3 -I Scripts/reference/dump_headers.py` after any revision bump.

Prequantized packs (8-bit, SHA-256 sidecars, `VincentGOURBIN/idol-mlx-packs`) are planned for Jalon 6 and published only after approval.
