// TinyDenoiserParityTests.swift - denoiser layers and a tiny LatentSeqMMFlowModel (fp32, tier 1)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import Testing
@testable import IdolCore

@Suite("TinyDenoiserParity", .serialized)
struct TinyDenoiserParityTests {
    typealias P = TinyParity
    static let tol: Float = 1e-4

    func load(_ m: Module, _ c: String) throws { try ModuleLoading.load(m, from: P.denoiser, prefix: "\(c).w.") }

    @Test func positionEmbedderV1() {
        let m = PcdPositionEmbedderV1(channels: 128)
        #expect(P.relMax(m(P.tensor("pos_v1.in.x")), P.tensor("pos_v1.out.y")) < Self.tol)
    }

    @Test func rePo() throws {
        let m = RePo3DRotaryEmbedding(modelChannels: 128, heads: 2)
        try load(m, "repo")
        let (c, s) = m(P.tensor("repo.in.x"))
        #expect(P.relMax(c, P.tensor("repo.out.cos")) < Self.tol)
        #expect(P.relMax(s, P.tensor("repo.out.sin")) < Self.tol)
    }

    @Test func interleavedRotary() {
        let y = applyInterleavedRotary(P.tensor("rotary.in.x"), cos: P.tensor("rotary.in.cos"), sin: P.tensor("rotary.in.sin"))
        #expect(P.relMax(y, P.tensor("rotary.out.y")) < 1e-6)
    }

    @Test func ropeAttention() throws {
        let m = RopeMultiHeadAttention(channels: 128, heads: 2)
        try load(m, "rope_attn")
        let y = m(P.tensor("rope_attn.in.x"), cos: P.tensor("rope_attn.in.cos"), sin: P.tensor("rope_attn.in.sin"))
        #expect(P.relMax(y, P.tensor("rope_attn.out.y")) < Self.tol)
    }

    @Test(arguments: [("unified_mod", true), ("unified_plain", false)])
    func unifiedBlock(name: String, modulated: Bool) throws {
        let m = UnifiedTransformerBlock(channels: 128, heads: 2, modulation: modulated)
        try load(m, name)
        let y = m(P.tensor("\(name).in.x"), mod: modulated ? P.tensor("\(name).in.mod") : nil,
                  cos: P.tensor("\(name).in.cos"), sin: P.tensor("\(name).in.sin"))
        #expect(P.relMax(y, P.tensor("\(name).out.y")) < Self.tol)
    }

    @Test func fullModel() throws {
        let m = IdolDenoiser(qTokenLength: 32, inChannels: 16, camChannels: 5, modelChannels: 128, condChannels: 40,
                             cond2Channels: 8, numBlocks: 2, numRefinerBlocks: 2, numHeads: 2, blockDType: .float32)
        try load(m, "dit")
        let v = m(latent: P.tensor("dit.in.latent"), camera: P.tensor("dit.in.camera"), t: P.tensor("dit.in.t"),
                  feature1: P.tensor("dit.in.feature1"), feature2: P.tensor("dit.in.feature2"))
        let rl = P.relMax(v.latent, P.tensor("dit.out.latent")), rc = P.relMax(v.camera, P.tensor("dit.out.camera"))
        print("TINYDIT latent \(rl) camera \(rc)")
        #expect(rl < Self.tol && rc < Self.tol)
    }
}
