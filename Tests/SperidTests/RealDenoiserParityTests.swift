// RealDenoiserParityTests.swift - the denoiser on the real weights and a real object (tier 2)
// Copyright 2026 Vincent Gourbin
// Needs $SPERID_MODELS_DIR/parity/denoiser_avocado.safetensors (Scripts/reference/denoiser_fixtures.py).

import Foundation
import MLX
import Testing
@testable import IdolCore
@testable import SperidCore

enum RealDenoiserFixture {
    static let url = RealFixtures.root.appendingPathComponent("parity/denoiser_avocado.safetensors")
    static var available: Bool {
        FileManager.default.fileExists(atPath: url.path)
            && FileManager.default.fileExists(atPath: IdolWeights.denoiser.localURL(in: RealFixtures.root).path)
    }
    nonisolated(unsafe) static let arrays: [String: MLXArray] = (try? loadArrays(url: url)) ?? [:]
}

@Suite("RealDenoiserParity", .serialized, .enabled(if: RealDenoiserFixture.available))
struct RealDenoiserParityTests {
    static func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        (sqrt(sum(square(a.asType(.float32) - b))) / sqrt(sum(square(b)))).item(Float.self)
    }

    static func forward(_ dtype: DType, negative: Bool) throws -> (latent: MLXArray, camera: MLXArray) {
        let model = try IdolDenoiser.load(modelsDirectory: RealFixtures.root, config: RealFixtures.config(), blockDType: dtype)
        let f = RealDenoiserFixture.arrays
        var c = IdolCondition(feature1: f["feature1"]!, feature2: f["feature2"]!)
        if negative { c = c.negative }
        return model(latent: f["latent_noise"]!, camera: f["camera_noise"]!, t: MLXArray([Float(1000)]),
                     feature1: c.feature1, feature2: c.feature2)
    }

    /// fp32 against fp32: the port itself, on the avocado's real conditioning (cond and neg branches).
    @Test(arguments: [false, true])
    func forwardFP32MatchesPython(negative: Bool) throws {
        let tag = negative ? "neg" : "cond"
        let v = try Self.forward(.float32, negative: negative)
        let f = RealDenoiserFixture.arrays
        let rl = Self.relL2(v.latent, f["v32_\(tag)_latent"]!), rc = Self.relL2(v.camera, f["v32_\(tag)_camera"]!)
        print("REALDIT fp32 \(tag) latent \(rl) camera \(rc)")
        // 30 blocks over 12 294 tokens, CPU (torch) against GPU (MLX) summation order: measured 7.7e-5
        // (cond) and 2.6e-4 (neg) on 2026-11-15, four times under the fp16 noise of the reference.
        #expect(rl < 5e-4 && rc < 5e-4)
    }

    /// fp16 (reference precision) within the reference's own fp16 noise (Python fp16 vs fp32).
    @Test(arguments: [false, true])
    func forwardFP16WithinReferenceNoise(negative: Bool) throws {
        let tag = negative ? "neg" : "cond"
        let v = try Self.forward(.float16, negative: negative)
        let f = RealDenoiserFixture.arrays
        let swift = Self.relL2(v.latent, f["v32_\(tag)_latent"]!)
        let python = Self.relL2(f["v16_\(tag)_latent"]!, f["v32_\(tag)_latent"]!)
        print("REALDIT fp16 \(tag) vs fp32: swift \(swift) python \(python)")
        #expect(swift <= max(1.5 * python, 1e-3))
    }
}
