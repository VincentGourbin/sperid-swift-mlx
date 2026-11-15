// ReplayNoise.swift - replay the draws recorded by the Python reference
// Copyright 2026 Vincent Gourbin

import MLX

/// Serves the tensors a fixture recorded: `<prefix>offsets_<level>`, `<prefix>jitter`,
/// `<prefix>latent_noise`, `<prefix>camera_noise`, `<prefix>vae_noise`. A missing tensor is a programming
/// error (fatal). A count that differs from the recording (the octree split one cell differently) is
/// *reported* in `mismatches` and served truncated or padded with 0.5, so a test can fail cleanly
/// instead of killing the test process.
public struct ReplayNoise: NoiseSource {
    let arrays: [String: MLXArray]
    let prefix: String
    public private(set) var mismatches: [String] = []

    public init(arrays: [String: MLXArray], prefix: String = "") {
        self.arrays = arrays
        self.prefix = prefix
    }

    func tensor(_ name: String) -> MLXArray {
        guard let a = arrays[prefix + name] else { fatalError("ReplayNoise: no recorded \(prefix + name)") }
        return a
    }

    public mutating func latentNoise(shape: [Int]) -> MLXArray { tensor("latent_noise").reshaped(shape) }
    public mutating func cameraNoise(shape: [Int]) -> MLXArray { tensor("camera_noise").reshaped(shape) }
    public mutating func vaePosterior(shape: [Int]) -> MLXArray { tensor("vae_noise").reshaped(shape) }

    public mutating func systematicOffsets(level: Int, count: Int) -> [Float] {
        fit(tensor("offsets_\(level)").asType(.float32).asArray(Float.self), count, "offsets_\(level)")
    }

    public mutating func cellJitter(count: Int) -> [Float] {
        fit(tensor("jitter").asType(.float32).asArray(Float.self), count, "jitter")
    }

    mutating func fit(_ values: [Float], _ count: Int, _ name: String) -> [Float] {
        guard values.count != count else { return values }
        mismatches.append("\(name): \(values.count) recorded, \(count) requested")
        return Array((values + [Float](repeating: 0.5, count: max(0, count - values.count))).prefix(count))
    }
}

/// Fresh draws from a seed: MLX's generator for tensors, a SplitMix64 stream for the CPU draws.
/// Deterministic for a given seed, but a different stream from torch's (pitfall: never use it for parity).
public struct SeededNoise: NoiseSource {
    var key: MLXArray
    var state: UInt64

    public init(seed: UInt64) {
        key = MLXRandom.key(seed)
        state = seed &+ 0x9E37_79B9_7F4A_7C15
    }

    mutating func nextKey() -> MLXArray {
        let parts = MLXRandom.split(key: key)
        key = parts.0
        return parts.1
    }

    mutating func uniform(_ count: Int) -> [Float] {
        (0..<count).map { _ in
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z ^= z >> 31
            return Float(z >> 40) / Float(1 << 24)  // 24 random bits → [0, 1)
        }
    }

    public mutating func latentNoise(shape: [Int]) -> MLXArray { MLXRandom.normal(shape, key: nextKey()) }
    public mutating func cameraNoise(shape: [Int]) -> MLXArray { MLXRandom.normal(shape, key: nextKey()) }
    public mutating func vaePosterior(shape: [Int]) -> MLXArray { MLXRandom.normal(shape, key: nextKey()) }
    public mutating func systematicOffsets(level: Int, count: Int) -> [Float] { uniform(count) }
    public mutating func cellJitter(count: Int) -> [Float] { uniform(count) }
}
