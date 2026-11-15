# Reference configurations (planned)

A reference configuration fixes **every** setting that matters, so the same seed gives the same splat and a comparable time on comparable hardware. Each field maps to a setting that already exists; until it is measured, a value is written "to measure". The standard is the one used by yue2-mlx-swift (`docs/References.md` there).

| Id | Weights | CFG | Residency | Steps | Who it is for | Time | Peak |
|---|---|---|---|---|---|---|---|
| `16bit-fast` | fp16/bf16 as the reference | batched (if attention stays fused, T-2.5) | everything resident | 50 | quality reference, Mac | to measure | to measure |
| `16bit-lean` | fp16/bf16 | sequential | per stage | 50 | Mac 16-24 GB | to measure | to measure |
| `8bit-fast` | denoiser + decoders + DINOv3 8-bit | batched | everything resident | 50 | Mac | to measure | to measure |
| `8bit-lean` | same | sequential | per stage | 50 | iPhone 8 GB | to measure | to measure |
| `turbo/16bit-fast` | fp16 + phase LoRAs | none | everything resident | 5 | fast Mac preview | to measure | to measure |
| `turbo/8bit-lean` | 8-bit + phase LoRAs | none | per stage | 5 | iPhone | to measure | to measure |

4-bit is measured in T-6.3 and expected to fail parity on the iterated denoiser (error compounds over 100 evaluations; YuE2's 4-bit flow stage was rejected for the same reason). BiRefNet and the FLUX.2 VAE encoder stay unquantized. Steps are identical inside a family: speed is never bought with quality inside a profile.
