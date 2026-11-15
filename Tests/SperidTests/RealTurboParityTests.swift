// RealTurboParityTests.swift - idol-turbo phase LoRAs on the avocado (tier 2)
// Copyright 2026 Vincent Gourbin
// Needs $SPERID_MODELS_DIR/parity/turbo_avocado.safetensors (Scripts/reference/turbo_fixtures.py).

import Foundation
import MLX
import MLXNN
import Testing
@testable import IdolCore
@testable import SperidCore

enum RealTurboFixture {
    static let url = RealFixtures.root.appendingPathComponent("parity/turbo_avocado.safetensors")
    static var available: Bool { FileManager.default.fileExists(atPath: url.path) && RealDenoiserFixture.available }
    nonisolated(unsafe) static let arrays: [String: MLXArray] = (try? loadArrays(url: url)) ?? [:]
}

@Suite("RealTurboParity", .serialized, .enabled(if: RealTurboFixture.available))
struct RealTurboParityTests {
    static func turboModel(_ dtype: DType) throws -> (IdolDenoiser, TurboPhaseState) {
        let m = try IdolDenoiser.load(modelsDirectory: RealFixtures.root, config: RealFixtures.config(), blockDType: dtype)
        let state = try TurboLoRA.inject(into: m, url: IdolWeights.turboLoRA.localURL(in: RealFixtures.root))
        return (m, state)
    }

    static func relL2(_ a: MLXArray, _ b: MLXArray) -> Float { RealDenoiserParityTests.relL2(a, b) }

    @Test func injectionReplacesTheLinears() throws {
        let (m, _) = try Self.turboModel(.float16)
        #expect(m.blocks[23].attn.qkv is PhaseLoRALinear)
        #expect(m.contextRefiner[1].mlp.mlp[2] is PhaseLoRALinear)
    }

    @Test(arguments: [DType.float32, DType.float16])
    func expertsMatch(dtype: DType) throws {
        let (m, state) = try Self.turboModel(dtype)
        let d = RealDenoiserFixture.arrays, f = RealTurboFixture.arrays
        for (phase, t) in [(TurboPhase.slow, Float(1000)), (TurboPhase.fast, Float(400))] {
            state.phase = phase
            let v = m(latent: d["latent_noise"]!, camera: d["camera_noise"]!, t: MLXArray([t]),
                      feature1: d["feature1"]!, feature2: d["feature2"]!)
            let ref = f["v32_\(phase.rawValue)_latent"]!
            let swift = Self.relL2(v.latent, ref), python = Self.relL2(f["v16_\(phase.rawValue)_latent"]!, ref)
            print("REALTURBO \(dtype) \(phase.rawValue) vs fp32: swift \(swift) python fp16 \(python)")
            if dtype == .float32 { #expect(swift < 5e-4) } else { #expect(swift <= max(1.5 * python, 1e-3)) }
        }
    }

    @Test func fiveStepSampleMatches() throws {
        let (m, state) = try Self.turboModel(.float16)
        let d = RealDenoiserFixture.arrays, f = RealTurboFixture.arrays
        let start = Date()
        let sample = try FlowSampler.sample(
            denoiser: m, latentNoise: d["latent_noise"]!, cameraNoise: d["camera_noise"]!,
            condition: IdolCondition(feature1: d["feature1"]!, feature2: d["feature2"]!),
            schedule: FlowSchedule(steps: 5, rescaleT: 3), cfgStrength: 0, turbo: state)
        let r = Self.relL2(sample.latent, f["turbo_latent"]!)
        print(String(format: "REALTURBO 5-step final latent relL2 %.6f camera %.6f in %.1f s (not a benchmark)",
                     r, Self.relL2(sample.camera, f["turbo_camera"]!), Date().timeIntervalSince(start)))
        #expect(r < 2e-2)
    }
}
