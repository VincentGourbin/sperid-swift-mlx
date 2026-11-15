# Third-party notices

This repository contains **no model weights**. Its Swift code is an independent port written from the
reference implementations below; where a Swift file follows the structure of one of them, the file header
names it. The original licences apply to that structure; the Swift code is released under the MIT licence
(see `LICENSE`) without changing the terms of the works it follows.

## Code followed by the port

| Work | Licence | Followed in |
|---|---|---|
| [speridlabs/idol-sdk](https://github.com/speridlabs/idol-sdk) (Speri Labs) — pipeline, denoiser, decoders, Gaussian output, preprocessing, BiRefNet wrapper | Apache-2.0 | `Sources/IdolCore` (whole pipeline), `Sources/IdolCore/Models/BiRefNet` |
| The denoiser of idol-sdk is itself adapted from [diffusers](https://github.com/huggingface/diffusers) `transformer_wan.py` (The Wan Team and The HuggingFace Team) | Apache-2.0 | `Models/Denoiser` |
| [huggingface/transformers](https://github.com/huggingface/transformers) — `modeling_dinov3_vit.py` | Apache-2.0 | `Models/Encoders/DINOv3.swift` |
| [huggingface/diffusers](https://github.com/huggingface/diffusers) — `AutoencoderKL` encoder blocks, `DiagonalGaussianDistribution` | Apache-2.0 | `Models/Encoders/FluxVAEEncoder.swift`, `FluxVAETokens.swift` |
| [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (Zheng et al.), as shipped with [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) | MIT | `Models/BiRefNet` |
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer) (Microsoft) | MIT | `Models/BiRefNet/Swin.swift` |
| [torchvision](https://github.com/pytorch/vision) `deform_conv2d` sampling rule | BSD-3-Clause | `Models/BiRefNet/Resample.swift` |
| [idol-turbo](https://huggingface.co/speridlabs/idol-turbo) `turbo.py` (phase LoRA sampling) | Apache-2.0 | `Models/Denoiser/TurboLoRA.swift`, `Sampling/FlowSampler.swift` |
| [yue2-mlx-swift](https://github.com/VincentGourbin/yue2-mlx-swift) (resumable downloader, scripts) | MIT | `SperidCore/Loading/HubDownloader.swift`, `Scripts/` |

Apache-2.0 text: https://www.apache.org/licenses/LICENSE-2.0 — the idol-sdk copyright notice is that of its
`LICENSE` file (Speri Labs).

## Model weights (downloaded at run time, never redistributed here)

| Weights | Published by | Licence (as stated on the repository) |
|---|---|---|
| `speridlabs/idol` (denoiser, decoders, BiRefNet) | Speri Labs | Apache-2.0 |
| `speridlabs/idol-turbo` (phase LoRAs) | Speri Labs | Apache-2.0 |
| `facebook/dinov3-vith16plus-pretrain-lvd1689m` | Meta | DINOv3 Licence — gated, terms accepted on Hugging Face |
| `black-forest-labs/FLUX.2-dev` (`vae/`, encoder only) | Black Forest Labs | FLUX Non-Commercial License — gated |

Whoever runs the code downloads these weights under their own account and is bound by their terms. Read
them before any commercial use; nothing here grants rights on the weights.
