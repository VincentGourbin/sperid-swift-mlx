// ConfigTests.swift - config.json decoding and derived sizes
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import IdolCore

@Suite("Config")
struct ConfigTests {
    static func fixture() throws -> IdolConfig {
        let url = try #require(Bundle.module.url(forResource: "idol-config", withExtension: "json",
                                                 subdirectory: "Fixtures"))
        return try IdolConfig.load(from: url)
    }

    @Test func decodesThePublishedConfig() throws {
        let c = try Self.fixture()
        #expect(c.denoiser.qTokenLength == 8192)
        #expect(c.denoiser.numBlocks == 24 && c.denoiser.numRefinerBlocks == 2)
        #expect(c.denoiser.condChannels == 1280 && c.denoiser.cond2Channels == 128)
        #expect(c.decoder.numBlocks == 4 && c.decoderGS.numBlocks == 16)
        #expect(c.decoderGS.representationConfig.numGaussians == 32)
        #expect(c.decoderGS.representationConfig.lr["_rotation"] == 0.1)
        #expect(c.decoderGS.representationConfig.filterKernelSize3D == 0.0009)
        #expect(c.imageSize == 1024 && c.maxVoxelLevel == 8)
        #expect(c.imageEncoders["dinov3"] == "facebook/dinov3-vith16plus-pretrain-lvd1689m")
    }

    @Test func tokenCounts() throws {
        let c = try Self.fixture()
        #expect(c.conditionTokenCount == 4101)  // CLS + 4 registers + 64² patches
        #expect(c.jointTokenCount == 12_294)
    }

    @Test func roundTripsThroughJSON() throws {
        let c = try Self.fixture()
        let again = try JSONDecoder().decode(IdolConfig.self, from: JSONEncoder().encode(c))
        #expect(again == c)
    }
}

@Suite("Request")
struct RequestTests {
    @Test func defaultsMatchTheSDK() throws {
        let r = IdolRequest()
        #expect(r.steps == 50 && r.cfgStrength == 3 && r.rescaleT == 3 && r.seed == 42)
        #expect(r.numGaussians == 262_144 && r.anchorCount() == 8192)
        #expect(r.effectiveCropMargin == 1.2)
        #expect(IdolRequest(mode: .real).effectiveCropMargin == 1.7)
        try r.validate()
    }

    @Test func turboIsGuidanceFree() {
        #expect(IdolRequest.turbo.steps == 5 && IdolRequest.turbo.cfgStrength == 0)
    }

    @Test func boundsAreEnforcedAndRoundingFollowsTheSDK() throws {
        #expect(throws: (any Error).self) { try IdolRequest(numGaussians: 16_384).validate() }
        #expect(throws: (any Error).self) { try IdolRequest(steps: 0).validate() }
        let odd = IdolRequest(numGaussians: 100_001)
        try odd.validate()
        #expect(odd.anchorCount() == 3125)  // 100_001 // 32
    }
}
