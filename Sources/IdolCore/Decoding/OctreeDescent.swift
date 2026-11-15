// OctreeDescent.swift - place anchor points by descending the octree (`OctreeProbabilityFixedlenDecoder.sample`)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// The descent runs on the CPU (data-dependent shapes): at each level the network gives each occupied
/// cell its 8 child probabilities, `SystematicSampler` splits the cell's points, empty children are
/// dropped and the rest compacted in order (parent-major, child i + 2j + 4k with x fastest, pitfall 13).
/// At the last level every point lands uniformly inside its leaf: (coord + jitter) / 2^level.
public enum OctreeDescent {
    /// Child offsets in the reference order: index = i + 2j + 4k.
    static let childOffsets: [(Int32, Int32, Int32)] = (0..<8).map { c in
        (Int32(c & 1), Int32((c >> 1) & 1), Int32((c >> 2) & 1))
    }

    /// Per-level statistics, for parity tests and logs.
    public struct LevelStats: Sendable, Equatable {
        public let level: Int
        public let cells: Int
    }

    /// - Parameters:
    ///   - probabilities: `(cellCentres [N·3] ∈ [0,1], resolution 1 << level) → N rows of 8 probabilities`.
    /// - Returns: anchors [numPoints·3] ∈ [0, 1] and the cell count per level.
    public static func descend<N: NoiseSource>(
        numPoints: Int, levels: Int, noise: inout N,
        probabilities: (_ centres: [Float], _ resolution: Int) throws -> [[Float]]
    ) rethrows -> (anchors: [Float], stats: [LevelStats]) {
        var coords: [(Int32, Int32, Int32)] = [(0, 0, 0)]
        var counts: [Int] = [numPoints]
        var stats: [LevelStats] = []
        for level in 1...levels {
            let parentResolution = Float(1 << (level - 1))
            var centres = [Float]()
            centres.reserveCapacity(coords.count * 3)
            for c in coords {
                centres += [(Float(c.0) + 0.5) / parentResolution, (Float(c.1) + 0.5) / parentResolution,
                            (Float(c.2) + 0.5) / parentResolution]
            }
            let probs = try probabilities(centres, 1 << level)
            precondition(probs.count == coords.count, "one probability row per cell")
            let offsets = noise.systematicOffsets(level: level, count: coords.count)
            var nextCoords: [(Int32, Int32, Int32)] = []
            var nextCounts: [Int] = []
            for (i, parent) in coords.enumerated() {
                let split = SystematicSampler.split(probabilities: probs[i], count: counts[i], offset: offsets[i])
                for (child, n) in split.enumerated() where n > 0 {
                    let o = childOffsets[child]
                    nextCoords.append((parent.0 * 2 + o.0, parent.1 * 2 + o.1, parent.2 * 2 + o.2))
                    nextCounts.append(n)
                }
            }
            assert(nextCounts.reduce(0, +) == numPoints, "systematic resampling must conserve the count")
            coords = nextCoords
            counts = nextCounts
            stats.append(LevelStats(level: level, cells: coords.count))
        }
        let jitter = noise.cellJitter(count: numPoints * 3)
        let leafResolution = Float(1 << levels)
        var anchors = [Float](repeating: 0, count: numPoints * 3)
        var p = 0
        for (c, n) in zip(coords, counts) {
            for _ in 0..<n {
                anchors[3 * p] = (Float(c.0) + jitter[3 * p]) / leafResolution
                anchors[3 * p + 1] = (Float(c.1) + jitter[3 * p + 1]) / leafResolution
                anchors[3 * p + 2] = (Float(c.2) + jitter[3 * p + 2]) / leafResolution
                p += 1
            }
        }
        return (anchors, stats)
    }

    /// The network path: `decoder` forward + softmax per level, on one object (B = 1).
    public static func descend<N: NoiseSource>(decoder: OctreeDecoder, latent: MLXArray, numPoints: Int, levels: Int,
                                               noise: inout N) -> (anchors: [Float], stats: [LevelStats]) {
        var gpuUs: UInt64 = 0
        let start = IdolProfiling.now()
        let result = descend(numPoints: numPoints, levels: levels, noise: &noise) { centres, resolution in
            let t0 = IdolProfiling.now()
            let n = centres.count / 3
            let x = MLXArray(centres, [1, n, 3])
            let logits = decoder(x, resolution: MLXArray([Float(resolution)]), latent: latent)
            let flat = softmax(logits.asType(.float32), axis: -1, precise: true).asArray(Float.self)
            gpuUs += (IdolProfiling.now() - t0) / 1000
            return (0..<n).map { Array(flat[(8 * $0)..<(8 * $0 + 8)]) }
        }
        let totalUs = (IdolProfiling.now() - start) / 1000
        IdolProfiling.complete("5a. octree forwards + readback (GPU)", .decoding, microseconds: gpuUs)
        IdolProfiling.complete("5b. octree split + compaction (CPU)", .decoding, microseconds: totalUs - gpuUs)
        return result
    }
}
