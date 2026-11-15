// NoiseSource.swift - every random draw of the pipeline, injectable for parity
// Copyright 2026 Vincent Gourbin

import MLX

/// MLX's generator cannot reproduce torch's stream, so parity runs replay noise dumped by the
/// Python reference. Every stochastic point of the pipeline goes through this protocol:
/// - `latentNoise` [1, 8192, 16] and `cameraNoise` [1, 1, 5] (sampler start),
/// - `vaePosterior` (the FLUX.2 VAE latent is *sampled*, `latent_dist.sample()`),
/// - `systematicOffsets` (one u0 ∈ [0, 1/n) per octree cell, per level),
/// - `cellJitter` [num_points, 3] (position of each anchor inside its leaf cell).
public protocol NoiseSource {
    mutating func latentNoise(shape: [Int]) -> MLXArray
    mutating func cameraNoise(shape: [Int]) -> MLXArray
    mutating func vaePosterior(shape: [Int]) -> MLXArray
    /// `count` uniform draws in [0, 1), scaled by the caller.
    mutating func systematicOffsets(level: Int, count: Int) -> [Float]
    mutating func cellJitter(count: Int) -> [Float]
}
