// TinyOctreeParityTests.swift - octree decoder and descent against the upstream classes (tier 1)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import IdolCore

@Suite("TinyOctreeParity", .serialized)
struct TinyOctreeParityTests {
    typealias P = TinyParity

    static func decoder() throws -> OctreeDecoder {
        let d = OctreeDecoder(modelChannels: 64, condChannels: 16, numBlocks: 2, numHeads: 4, blockDType: .float32)
        try ModuleLoading.load(d, from: P.arrays, prefix: "octree.w.")
        return d
    }

    @Test func forwardMatches() throws {
        let d = try Self.decoder()
        let logits = d(P.tensor("octree.in.x"), resolution: P.tensor("octree.in.level"), latent: P.tensor("octree.in.latent"))
        #expect(P.relMax(logits, P.tensor("octree.out.logits")) < 1e-4)
    }

    /// With the reference's own probabilities and draws, the descent is exact: integer cells, the
    /// same systematic split, the same jitter.
    @Test func descentWithReferenceProbabilitiesIsBitExact() {
        var noise = ReplayNoise(arrays: P.arrays, prefix: "descent.out.")
        var level = 0
        let result = OctreeDescent.descend(numPoints: 256, levels: 4, noise: &noise) { centres, resolution in
            level += 1
            #expect(resolution == 1 << level)
            let flat = P.tensor("descent.out.probs_\(level)").asArray(Float.self)
            #expect(flat.count == centres.count / 3 * 8)
            return (0..<(flat.count / 8)).map { Array(flat[(8 * $0)..<(8 * $0 + 8)]) }
        }
        let expected = P.tensor("descent.out.anchors").asArray(Float.self)
        #expect(result.anchors == expected)
        // cells after levels 1-3 = the number of offsets recorded for levels 2-4
        let recorded = (2...4).map { P.tensor("descent.out.offsets_\($0)").size }
        #expect(Array(result.stats.prefix(3).map(\.cells)) == recorded)
    }

    /// The network path end to end: Swift probabilities, recorded draws.
    @Test func descentWithSwiftNetworkMatches() throws {
        let d = try Self.decoder()
        var noise = ReplayNoise(arrays: P.arrays, prefix: "descent.out.")
        let result = OctreeDescent.descend(decoder: d, latent: P.tensor("octree.in.latent"), numPoints: 256,
                                           levels: 4, noise: &noise)
        let expected = P.tensor("descent.out.anchors").asArray(Float.self)
        let maxDiff = zip(result.anchors, expected).map { abs($0 - $1) }.max() ?? 0
        #expect(maxDiff < 1e-6, "anchors differ by \(maxDiff): a cell split differently")
    }
}
