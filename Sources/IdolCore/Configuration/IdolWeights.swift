// IdolWeights.swift - the pinned weight files and their loader contracts
// Copyright 2026 Vincent Gourbin

import Foundation
import SperidCore

/// Every file the Idol pipeline reads, pinned to a Hub revision with its SHA-256 (Hub tree API,
/// 2026-11-15). Tensor counts and anchor shapes come from the published headers
/// (`Scripts/reference/dump_headers.py`); see `plan/02.6-tenseurs.md`.
public enum IdolWeights {
    public static let idolRepo = "speridlabs/idol"
    public static let idolRevision = "2731c2291fc65a797c5cb627a5e5526fc1490902"
    public static let turboRepo = "speridlabs/idol-turbo"
    public static let turboRevision = "45cdd41f1f4d6d70496daa7d619f57ad79307a37"
    /// ComfyUI repack: the only ungated copy of the DINOv3 ViT-H+/16 and FLUX.2 VAE weights.
    /// Not used by default: see `docs/knowledge/decisions/weights-source.md` (licences).
    public static let comfyRepo = "Comfy-Org/Idol"
    public static let comfyRevision = "e907d8dca6c465735eaf4ce9b3a3c2c6401d5705"

    public static let config = HubFile(repoID: idolRepo, revision: idolRevision, path: "config.json",
                                       size: 1_335, sha256: nil)
    public static let denoiser = HubFile(
        repoID: idolRepo, revision: idolRevision, path: "denoiser.safetensors", size: 1_482_258_308,
        sha256: "ded2cacf671bf3978ebc62729087abdbd318ef4752e6805b4c9fc8ffa51fe716")
    public static let octreeDecoder = HubFile(
        repoID: idolRepo, revision: idolRevision, path: "decoder.safetensors", size: 203_171_696,
        sha256: "9da4d7676bfcadb1c3fdfbff796e6257dadf82d26ec05be7717c79b2b13a1c4e")
    public static let gaussianDecoder = HubFile(
        repoID: idolRepo, revision: idolRevision, path: "decoder_gs.safetensors", size: 949_083_036,
        sha256: "3087eb8ee5763206a53e4078acf2a1bf382dab5147d407ec13a2cf39c04cc544")
    public static let birefnet = HubFile(
        repoID: idolRepo, revision: idolRevision, path: "birefnet.safetensors", size: 444_466_856,
        sha256: "44bbc97b041f879975d0d6d538b8e66a25eb04d1f49aab40697544cdd97eaa8e")
    /// v5 phase-LoRA experts (slow for t ≥ 0.5, fast below), rank 32, alpha 128.
    public static let turboLoRA = HubFile(
        repoID: turboRepo, revision: turboRevision, path: "idol_turbo_lora.safetensors", size: 117_484_528,
        sha256: "2e4b089211c5d79354f8b67d512f9b436248ac072cea171f56d393a248f7bf66")
    public static let dinov3Comfy = HubFile(
        repoID: comfyRepo, revision: comfyRevision, path: "clip_vision/dino_v3_vit_h_bf16.safetensors",
        size: 1_681_247_696, sha256: "a29ef35101a16966972a0d50732a6f3a608ff7cfffb2afa9bbe9007cb842cc53")
    public static let flux2VAEComfy = HubFile(
        repoID: comfyRepo, revision: comfyRevision, path: "vae/flux2-vae.safetensors",
        size: 336_211_292, sha256: "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3")

    /// DINOv3 ViT-H+/16, official repository (gated: accept Meta's DINOv3 licence, then HF_TOKEN).
    public static let dinov3 = HubFile(
        repoID: "facebook/dinov3-vith16plus-pretrain-lvd1689m", revision: "c807c9eeea853df70aec4069e6f56b28ddc82acc",
        path: "model.safetensors", size: 3_362_432_800,
        sha256: "3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d", gated: true)
    /// The FLUX.2 VAE (only its encoder is used), official repository (gated: accept the FLUX.2-dev terms).
    public static let flux2VAE = HubFile(
        repoID: "black-forest-labs/FLUX.2-dev", revision: "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
        path: "vae/diffusion_pytorch_model.safetensors", size: 336_213_556,
        sha256: "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5", gated: true)
    public static let encoderFiles: [HubFile] = [dinov3, flux2VAE]

    /// The files `speridlabs/idol` itself publishes (≈ 3.1 GB, Apache-2.0).
    public static let idolFiles: [HubFile] = [config, denoiser, octreeDecoder, gaussianDecoder, birefnet]

    // MARK: - Contracts (from the published headers)

    public static let denoiserContract = WeightContract(file: denoiser.path, tensorCount: 560, anchors: [
        "pos_pe": [1, 8192, 3],
        "input_layer.weight": [1024, 16],
        "cond_embedder.weight": [1024, 1280],
        "cond_embedder2.weight": [1024, 128],
        "blocks.23.attn.qkv.weight": [3072, 1024],
        "repo_layers.23.freqs_2": [12],
        "shift_table": [1, 2, 1024],
        "cam_out_layer.weight": [5, 1024],
    ])
    public static let octreeDecoderContract = WeightContract(file: octreeDecoder.path, tensorCount: 60, anchors: [
        "in_proj.weight": [1024, 3],
        "blocks.3.cross_attn.to_kv.weight": [2048, 16],
        "out_proj.weight": [8, 1024],
    ])
    public static let gaussianDecoderContract = WeightContract(file: gaussianDecoder.path, tensorCount: 328, anchors: [
        "blocks.15.self_attn.to_qkv.weight": [3072, 1024],
        "out_proj.weight": [480, 1024],
        "points_offset_perturbation": [32, 3],
        "base_offset_scale": [],
    ])
    public static let birefnetContract = WeightContract(file: birefnet.path, tensorCount: 742, anchors: [:])
    public static let turboLoRAContract = WeightContract(file: turboLoRA.path, tensorCount: 448, anchors: [
        "blocks.0.attn.qkv.slow_A": [32, 1024],
        "blocks.0.attn.qkv.slow_B": [3072, 32],
    ])
    public static let dinov3Contract = WeightContract(file: dinov3Comfy.path, tensorCount: 615, anchors: [
        "embeddings.patch_embeddings.weight": [1280, 3, 16, 16],
        "embeddings.register_tokens": [1, 4, 1280],
        "layer.31.mlp.gate_proj.weight": [5120, 1280],
    ])
}
