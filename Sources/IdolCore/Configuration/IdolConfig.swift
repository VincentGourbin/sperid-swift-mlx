// IdolConfig.swift - speridlabs/idol config.json, decoded as published
// Copyright 2026 Vincent Gourbin

import Foundation

/// `config.json` of `speridlabs/idol` (identical in `idol-turbo`). Field names follow the file;
/// the Python constructors that consume each block are named in `plan/02.1-pipeline.md`.
public struct IdolConfig: Codable, Sendable, Equatable {
    public struct Denoiser: Codable, Sendable, Equatable {
        public let qTokenLength: Int
        public let inChannels: Int
        public let camChannels: Int
        public let modelChannels: Int
        public let condChannels: Int
        public let cond2Channels: Int
        public let numBlocks: Int
        public let numRefinerBlocks: Int
        public let numHeads: Int
        public let mlpRatio: Int

        enum CodingKeys: String, CodingKey {
            case qTokenLength = "q_token_length", inChannels = "in_channels", camChannels = "cam_channels"
            case modelChannels = "model_channels", condChannels = "cond_channels", cond2Channels = "cond2_channels"
            case numBlocks = "num_blocks", numRefinerBlocks = "num_refiner_blocks", numHeads = "num_heads"
            case mlpRatio = "mlp_ratio"
        }
    }

    public struct OctreeDecoder: Codable, Sendable, Equatable {
        public let modelChannels: Int
        public let condChannels: Int
        public let numBlocks: Int
        public let numHeads: Int
        public let mlpRatio: Int

        enum CodingKeys: String, CodingKey {
            case modelChannels = "model_channels", condChannels = "cond_channels", numBlocks = "num_blocks"
            case numHeads = "num_heads", mlpRatio = "mlp_ratio"
        }
    }

    public struct Representation: Codable, Sendable, Equatable {
        /// Gaussians per anchor point (32).
        public let numGaussians: Int
        /// Per-attribute multipliers on the raw decoder outputs (`_xyz`, `_features_dc`, …).
        public let lr: [String: Double]
        public let perturbeSize: Double
        public let filterKernelSize3D: Double
        public let scalingBias: Double
        public let opacityBias: Double

        enum CodingKeys: String, CodingKey {
            case numGaussians = "num_gaussians", lr, perturbeSize = "perturbe_size"
            case filterKernelSize3D = "3d_filter_kernel_size", scalingBias = "scaling_bias"
            case opacityBias = "opacity_bias"
        }
    }

    public struct GaussianDecoder: Codable, Sendable, Equatable {
        public let inChannels: Int
        public let modelChannels: Int
        public let condChannels: Int
        public let numBlocks: Int
        public let numHeads: Int
        public let mlpRatio: Int
        public let representationConfig: Representation

        enum CodingKeys: String, CodingKey {
            case inChannels = "in_channels", modelChannels = "model_channels", condChannels = "cond_channels"
            case numBlocks = "num_blocks", numHeads = "num_heads", mlpRatio = "mlp_ratio"
            case representationConfig = "representation_config"
        }
    }

    struct Named<Args: Codable & Sendable & Equatable>: Codable, Sendable, Equatable {
        let name: String
        let args: Args
    }

    public let denoiser: Denoiser
    public let decoder: OctreeDecoder
    public let decoderGS: GaussianDecoder
    public let imageSize: Int
    public let maxVoxelLevel: Int
    /// Encoder name (`dinov3`, `flux2_vae`) → Hub repo id.
    public let imageEncoders: [String: String]

    enum CodingKeys: String, CodingKey {
        case denoiser, decoder, decoderGS = "decoder_gs", imageSize = "image_size"
        case maxVoxelLevel = "max_voxel_level", imageEncoders = "image_encoders"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        denoiser = try c.decode(Named<Denoiser>.self, forKey: .denoiser).args
        self.decoder = try c.decode(Named<OctreeDecoder>.self, forKey: .decoder).args
        decoderGS = try c.decode(Named<GaussianDecoder>.self, forKey: .decoderGS).args
        imageSize = try c.decode(Int.self, forKey: .imageSize)
        maxVoxelLevel = try c.decode(Int.self, forKey: .maxVoxelLevel)
        imageEncoders = try c.decode([String: String].self, forKey: .imageEncoders)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(Named(name: "LatentSeqMMFlowModel", args: denoiser), forKey: .denoiser)
        try c.encode(Named(name: "OctreeProbabilityFixedlenDecoder", args: decoder), forKey: .decoder)
        try c.encode(Named(name: "GaussianFixedlenDecoder", args: decoderGS), forKey: .decoderGS)
        try c.encode(imageSize, forKey: .imageSize)
        try c.encode(maxVoxelLevel, forKey: .maxVoxelLevel)
        try c.encode(imageEncoders, forKey: .imageEncoders)
    }

    public static func load(from url: URL) throws -> IdolConfig {
        try JSONDecoder().decode(IdolConfig.self, from: Data(contentsOf: url))
    }

    /// Image tokens seen by the denoiser: DINOv3 CLS + 4 registers + (size/16)² patches; the
    /// FLUX.2 VAE stream is padded with 5 zero rows to the same length.
    public var conditionTokenCount: Int { 5 + (imageSize / 16) * (imageSize / 16) }

    /// Tokens of one joint denoiser block: latent + image + camera (8192 + 4101 + 1).
    public var jointTokenCount: Int { denoiser.qTokenLength + conditionTokenCount + 1 }
}
