// TinyLayersParityTests.swift - decoder layers against the upstream classes (fp32, tier 1)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import Testing
@testable import IdolCore

@Suite("TinyLayersParity", .serialized)
struct TinyLayersParityTests {
    typealias P = TinyParity
    static let tol: Float = 1e-5

    func load(_ m: Module, _ case_: String) throws { try ModuleLoading.load(m, from: P.arrays, prefix: "\(case_).w.") }

    @Test func layerNorms() throws {
        let affine = LayerNorm32(64, affine: true, eps: 1e-5)
        try load(affine, "ln_affine")
        #expect(P.relMax(affine(P.tensor("ln_affine.in.x")), P.tensor("ln_affine.out.y")) < Self.tol)
        let plain = LayerNorm32(64, affine: false, eps: 1e-6)
        #expect(P.relMax(plain(P.tensor("ln_plain.in.x")), P.tensor("ln_plain.out.y")) < Self.tol)
    }

    @Test func multiHeadRMSNorm() throws {
        let m = MultiHeadRMSNorm(dim: 16, heads: 4)
        try load(m, "rmsnorm")
        #expect(P.relMax(m(P.tensor("rmsnorm.in.x")), P.tensor("rmsnorm.out.y")) < Self.tol)
    }

    @Test func feedForward() throws {
        let m = FeedForwardNet(channels: 64)
        try load(m, "ffn")
        #expect(P.relMax(m(P.tensor("ffn.in.x")), P.tensor("ffn.out.y")) < Self.tol)
    }

    @Test func positionEmbedderV2() {
        let m = PcdPositionEmbedderV2(channels: 64)
        #expect(P.relMax(m(P.tensor("pos_v2.in.x")), P.tensor("pos_v2.out.y")) < Self.tol)
    }

    /// torch's fp32 `exp` is not correctly rounded (13-15 of 128 frequencies differ by one ulp from
    /// a correctly rounded exp, measured 2026-11-15); at arguments up to ~1600 rad one ulp of the
    /// angle is 1.2e-4. Neither side is "more right": the tolerance is that of fp32 trig (pitfall 20).
    static let trigTol: Float = 3e-4

    @Test func sinusoidalEmbedders() throws {
        let time = SinusoidalEmbedder(hidden: 64)
        try load(time, "time_embed")
        #expect(P.relMax(time(P.tensor("time_embed.in.t")), P.tensor("time_embed.out.y")) < Self.trigTol)
        let level = SinusoidalEmbedder(hidden: 64, maxPeriod: 1024, twoPi: true)
        try load(level, "level_embed")
        #expect(P.relMax(level(P.tensor("level_embed.in.t")), P.tensor("level_embed.out.y")) < Self.trigTol)
    }

    @Test func attention() throws {
        let s = MultiHeadAttention(channels: 64, heads: 4)
        try load(s, "self_attn")
        #expect(P.relMax(s(P.tensor("self_attn.in.x")), P.tensor("self_attn.out.y")) < Self.tol)
        let c = MultiHeadAttention(channels: 64, heads: 4, contextChannels: 16)
        try load(c, "cross_attn")
        let y = c(P.tensor("cross_attn.in.x"), context: P.tensor("cross_attn.in.context"))
        #expect(P.relMax(y, P.tensor("cross_attn.out.y")) < Self.tol)
    }

    @Test func blocks() throws {
        let a = ModulatedCrossOnlyBlock(channels: 64, contextChannels: 16, heads: 4)
        try load(a, "cross_only_block")
        let ya = a(P.tensor("cross_only_block.in.x"), mod: P.tensor("cross_only_block.in.mod"),
                   context: P.tensor("cross_only_block.in.context"))
        #expect(P.relMax(ya, P.tensor("cross_only_block.out.y")) < Self.tol)
        let b = TransformerCrossBlock(channels: 64, contextChannels: 16, heads: 4)
        try load(b, "cross_block")
        let yb = b(P.tensor("cross_block.in.x"), context: P.tensor("cross_block.in.context"))
        #expect(P.relMax(yb, P.tensor("cross_block.out.y")) < Self.tol)
    }
}
