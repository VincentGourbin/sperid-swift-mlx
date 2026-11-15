// TinyGaussianParityTests.swift - Gaussian decoder and assembly against the upstream classes (tier 1)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import IdolCore

@Suite("TinyGaussianParity", .serialized)
struct TinyGaussianParityTests {
    typealias P = TinyParity
    static let rep = IdolConfig.Representation(
        numGaussians: 32, lr: ["_xyz": 1, "_features_dc": 1, "_opacity": 1, "_scaling": 1, "_rotation": 0.1],
        perturbeSize: 1.5, filterKernelSize3D: 0.0009, scalingBias: 0.004, opacityBias: 0.1)

    static func decoder() throws -> GaussianDecoder {
        let d = GaussianDecoder(inChannels: 3, modelChannels: 64, condChannels: 16, numBlocks: 2, numHeads: 4,
                                representation: rep, blockDType: .float32)
        try ModuleLoading.load(d, from: P.arrays, prefix: "gaussian.w.")
        return d
    }

    @Test func forwardAndAssemblyMatch() throws {
        let d = try Self.decoder()
        let points = P.tensor("gaussian.in.points")
        let features = d(points, latent: P.tensor("gaussian.in.latent"))
        #expect(P.relMax(features, P.tensor("gaussian.out.features")) < 1e-4)
        let cloud = GaussianAssembly.assemble(anchors: points, features: P.tensor("gaussian.out.features"), decoder: d)
        for (name, values) in [("xyz", cloud.xyz), ("features_dc", cloud.featuresDC), ("scaling", cloud.scaling),
                               ("rotation", cloud.rotation), ("opacity", cloud.opacity)] {
            let expected = P.tensor("gaussian.out.\(name)").reshaped(-1)
            #expect(P.relMax(MLXArray(values), expected) < 1e-5, "\(name)")
        }
    }
}
