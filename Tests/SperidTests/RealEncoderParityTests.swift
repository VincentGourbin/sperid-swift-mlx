// RealEncoderParityTests.swift - DINOv3 and the FLUX.2 VAE encoder on the avocado (tier 2)
// Copyright 2026 Vincent Gourbin
// Needs $SPERID_MODELS_DIR/parity/encoders_avocado.safetensors (Scripts/reference/encoder_fixtures.py).

import Foundation
import MLX
import Testing
@testable import IdolCore
@testable import SperidCore

enum RealEncoderFixture {
    static let url = RealFixtures.root.appendingPathComponent("parity/encoders_avocado.safetensors")
    static var available: Bool { FileManager.default.fileExists(atPath: url.path) }
    nonisolated(unsafe) static let arrays: [String: MLXArray] = (try? loadArrays(url: url)) ?? [:]
}

@Suite("RealEncoderParity", .serialized, .enabled(if: RealEncoderFixture.available))
struct RealEncoderParityTests {
    static func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        (sqrt(sum(square(a.asType(.float32) - b))) / sqrt(sum(square(b)))).item(Float.self)
    }

    /// Mean per-token cosine similarity (plan/10-tests: ≥ 0.999 for DINOv3 in bf16).
    static func meanCosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let a = a.asType(.float32), b = b.asType(.float32)
        let num = sum(a * b, axis: -1)
        let den = sqrt(sum(a * a, axis: -1)) * sqrt(sum(b * b, axis: -1))
        return mean(num / maximum(den, MLXArray(Float(1e-12)))).item(Float.self)
    }

    @Test(arguments: [DType.float32, DType.bfloat16])
    func dinov3Matches(dtype: DType) throws {
        let model = try DINOv3.load(url: EncoderWeights.dinov3(modelsDirectory: RealFixtures.root), computeDType: dtype)
        let f = RealEncoderFixture.arrays
        let tokens = model.tokens(f["pixels"]!)
        let ref32 = f["dino32_tokens"]!
        let r = Self.relL2(tokens, ref32), cosine = Self.meanCosine(tokens, ref32)
        let python = Self.relL2(f["feature1_bf16"]!, ref32)
        print("REALENC dinov3 \(dtype) vs fp32: relL2 \(r) cos \(cosine) | python bf16 relL2 \(python)")
        if dtype == .float32 { #expect(r < 1e-3) } else { #expect(r <= 1.5 * python && cosine >= 0.999) }
    }

    @Test(arguments: [DType.float32, DType.bfloat16])
    func fluxVAEMatches(dtype: DType) throws {
        let vae = try FluxVAEEncoder.load(url: EncoderWeights.flux2VAE(modelsDirectory: RealFixtures.root), computeDType: dtype)
        let f = RealEncoderFixture.arrays
        let (meanNHWC, logvarNHWC) = vae.moments(f["pixels"]!)
        let rm = Self.relL2(meanNHWC.transposed(0, 3, 1, 2), f["vae32_mean"]!)
        let rv = Self.relL2(logvarNHWC.transposed(0, 3, 1, 2), f["vae32_logvar"]!)
        let tokens = vae.tokens(f["pixels"]!, noise: f["vae_noise"]!)
        let rt = Self.relL2(tokens, f["vae32_feature2"]!)
        let python = Self.relL2(f["feature2_bf16"]!, f["vae32_feature2"]!)
        print("REALENC vae \(dtype) vs fp32: mean \(rm) logvar \(rv) feature2 \(rt) | python bf16 feature2 \(python)")
        if dtype == .float32 { #expect(rm < 1e-3 && rt < 1e-3) } else { #expect(rt <= max(1.5 * python, 1e-2)) }
    }
}
