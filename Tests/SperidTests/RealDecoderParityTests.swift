// RealDecoderParityTests.swift - decoders on the real weights against the Python fixtures (tier 2)
// Copyright 2026 Vincent Gourbin
// Needs $SPERID_MODELS_DIR (or the default directory) with speridlabs/idol and parity/decoders.safetensors
// (Scripts/reference/real_fixtures.py). Skipped otherwise.

import Foundation
import MLX
import Testing
@testable import IdolCore
@testable import SperidCore

enum RealFixtures {
    static let root = ModelsDirectory.resolve()
    static let fixtureURL = root.appendingPathComponent("parity/decoders.safetensors")
    static var available: Bool {
        FileManager.default.fileExists(atPath: fixtureURL.path)
            && FileManager.default.fileExists(atPath: IdolWeights.octreeDecoder.localURL(in: root).path)
    }
    nonisolated(unsafe) static let arrays: [String: MLXArray] = (try? loadArrays(url: fixtureURL)) ?? [:]
    static func config() throws -> IdolConfig { try IdolConfig.load(from: IdolWeights.config.localURL(in: root)) }
}

@Suite("RealDecoderParity", .serialized, .enabled(if: RealFixtures.available))
struct RealDecoderParityTests {
    static func octree(_ blockDType: DType) throws -> OctreeDecoder {
        let d = OctreeDecoder(config: try RealFixtures.config().decoder, blockDType: blockDType)
        let weights = try loadArrays(url: IdolWeights.octreeDecoder.localURL(in: RealFixtures.root))
        try ModuleLoading.load(d, from: weights, dtype: ModuleLoading.referencePrecision(blockDType: blockDType))
        return d
    }

    static func probabilities(_ d: OctreeDecoder, _ cells: MLXArray, resolution: Int) -> MLXArray {
        let n = cells.dim(0)
        let logits = d(cells.reshaped(1, n, 3), resolution: MLXArray([Float(resolution)]), latent: RealFixtures.arrays["latent"]!)
        return softmax(logits.asType(.float32), axis: -1, precise: true).reshaped(n, 8)
    }

    /// Occupancy of an anchor list on a grid of `resolution`³ cells, as a multiset.
    static func leaves(_ anchors: [Float], resolution: Int = 256) -> [Int: Int] {
        var m: [Int: Int] = [:]
        for p in 0..<(anchors.count / 3) {
            let c = (0..<3).map { min(resolution - 1, Int(anchors[3 * p + $0] * Float(resolution))) }
            m[(c[0] << 16) | (c[1] << 8) | c[2], default: 0] += 1
        }
        return m
    }

    static func overlap(_ a: [Int: Int], _ b: [Int: Int]) -> Double {
        let shared = a.reduce(0) { $0 + min($1.value, b[$1.key] ?? 0) }
        return Double(shared) / Double(a.values.reduce(0, +))
    }

    /// The network level by level on the reference's own cells (teacher forcing): no cascade.
    @Test(arguments: [("fp16", DType.float16), ("fp32", DType.float32)])
    func levelProbabilitiesMatch(name: String, dtype: DType) throws {
        let d = try Self.octree(dtype)
        let f = RealFixtures.arrays
        let prefix = dtype == .float32 ? "fp32." : ""
        var diffs: [Float] = []
        for level in 1...8 {
            let swift = Self.probabilities(d, f["\(prefix)cells_\(level)"]!, resolution: 1 << level)
            diffs.append(abs(swift - f["\(prefix)probs_\(level)"]!).max().item(Float.self))
        }
        print("REALOCTREE \(name) max |Δp| per level \(diffs)")
        #expect(diffs.max()! < (dtype == .float32 ? 1e-4 : 2e-2))
    }

    /// fp32 end to end: the descent itself, with the recorded draws, against the fp32 reference.
    @Test func fp32DescentMatchesAnchors() throws {
        let d = try Self.octree(.float32)
        let f = RealFixtures.arrays
        let expected = f["fp32.anchors"]!.asArray(Float.self)
        var noise = ReplayNoise(arrays: f, prefix: "fp32.")
        let r = OctreeDescent.descend(decoder: d, latent: f["latent"]!, numPoints: expected.count / 3, levels: 8, noise: &noise)
        let same = Self.overlap(Self.leaves(r.anchors), Self.leaves(expected))
        let exact = r.anchors == expected
        print("REALOCTREE fp32 cells \(r.stats.map(\.cells)) exact \(exact) leafOverlap \(same) "
              + "mismatches \(noise.mismatches)")
        #expect(noise.mismatches.isEmpty)
        #expect(same >= 0.999)
    }

    /// fp16 (reference precision): cells may split differently from level 2 on, so the comparison is
    /// statistical — occupancy overlap on coarse grids (16³, 32³). Fresh draws: the reference's offsets
    /// no longer line up once a cell count differs, so fine leaves are not comparable.
    @Test func fp16DescentOccupiesTheSameLeaves() throws {
        let d = try Self.octree(.float16)
        let f = RealFixtures.arrays
        let expected = f["anchors"]!.asArray(Float.self)
        var noise = SeededNoise(seed: 0)
        let r = OctreeDescent.descend(decoder: d, latent: f["latent"]!, numPoints: expected.count / 3, levels: 8, noise: &noise)
        let o16 = Self.overlap(Self.leaves(r.anchors, resolution: 16), Self.leaves(expected, resolution: 16))
        let o32 = Self.overlap(Self.leaves(r.anchors, resolution: 32), Self.leaves(expected, resolution: 32))
        print("REALOCTREE fp16 cells \(r.stats.map(\.cells)) overlap16 \(o16) overlap32 \(o32)")
        #expect(o16 >= 0.97)
    }
}
