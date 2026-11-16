// IdolCore.swift - speridlabs/idol for Apple Silicon: one image → 3D Gaussian splat
// Copyright 2026 Vincent Gourbin

/// Port of `speridlabs/idol` (Apache-2.0, SDK `speridlabs/idol-sdk`). The pipeline, stage by stage
/// (`plan/02.1-pipeline.md`):
///
/// 1. preprocess — BiRefNet alpha (skipped for images with real alpha), 1-px erosion, square crop,
///    1024² composite on black (`Pipeline/`, Jalon 4);
/// 2. condition — DINOv3 ViT-H+/16 tokens [1, 4101, 1280] + FLUX.2 VAE tokens [1, 4101, 128]
///    (`Models/Encoders`, Jalon 3);
/// 3. denoise — `LatentSeqMMFlowModel`, 24 joint blocks over 8192 + 4101 + 1 tokens, Euler with
///    CFG (`Models/Denoiser`, `Sampling/`, Jalon 2) or the turbo phase LoRAs (Jalon 5);
/// 4. decode — octree descent to level 8 (`Decoding/`, CPU + 8 forwards), then 32 Gaussians per
///    anchor (`Models/Decoders`, Jalon 1);
/// 5. write — activations, `.ply`, camera (`Output/`).
public enum IdolCore {
    public static let version = "0.3.2"
}
