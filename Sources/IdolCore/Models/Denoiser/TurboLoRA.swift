// TurboLoRA.swift - idol-turbo's phase LoRA experts (`turbo.py`, `LoRALinear`)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import SperidCore

/// Which expert the sampler has selected (shared by every LoRA layer of one denoiser). `nil` = base.
public final class TurboPhaseState: @unchecked Sendable {
    public var phase: TurboPhase?
    public init() {}
}

/// `LoRALinear` of `turbo.py`: frozen base + two switchable rank-32 experts, `y = base(x) + (x·Aᵀ·Bᵀ)·α/r`,
/// the delta computed in fp32 (the experts' dtype) and cast to the output's dtype.
final class PhaseLoRALinear: Linear {
    let slow: (a: MLXArray, b: MLXArray)
    let fast: (a: MLXArray, b: MLXArray)
    let scale: Float
    let state: TurboPhaseState

    init(base: Linear, slow: (MLXArray, MLXArray), fast: (MLXArray, MLXArray), scale: Float, state: TurboPhaseState) {
        self.slow = slow
        self.fast = fast
        self.scale = scale
        self.state = state
        super.init(weight: base.weight, bias: base.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x)
        guard let phase = state.phase else { return y }
        let (a, b) = phase == .slow ? slow : fast
        let delta = matmul(matmul(x.asType(.float32), a.transposed()), b.transposed()) * scale
        return y + delta.asType(y.dtype)
    }
}

public enum TurboLoRA {
    public static let rank = 32
    public static let alpha: Float = 128

    /// Injects the experts of `idol_turbo_lora.safetensors` (448 tensors) into `attn.qkv`, `attn.out`,
    /// `mlp.mlp.0`, `mlp.mlp.2` of every block of `noise_refiner`, `context_refiner` and `blocks`.
    /// Returns the shared phase switch (starts at `.slow`).
    @discardableResult
    public static func inject(into denoiser: IdolDenoiser, url: URL) throws -> TurboPhaseState {
        try IdolWeights.turboLoRAContract.verify(SafetensorsHeader(contentsOf: url))
        let w = try loadArrays(url: url)
        let state = TurboPhaseState()
        state.phase = .slow
        let scale = alpha / Float(rank)
        func experts(_ prefix: String) throws -> ((MLXArray, MLXArray), (MLXArray, MLXArray)) {
            func t(_ s: String) throws -> MLXArray {
                guard let a = w["\(prefix).\(s)"] else { throw SperidError.weightMismatch("turbo LoRA: missing \(prefix).\(s)") }
                return a.asType(.float32)
            }
            return ((try t("slow_A"), try t("slow_B")), (try t("fast_A"), try t("fast_B")))
        }
        func wrap(_ base: Linear, _ prefix: String) throws -> PhaseLoRALinear {
            let (s, f) = try experts(prefix)
            return PhaseLoRALinear(base: base, slow: s, fast: f, scale: scale, state: state)
        }
        let groups: [(String, [UnifiedTransformerBlock])] = [
            ("noise_refiner", denoiser.noiseRefiner), ("context_refiner", denoiser.contextRefiner), ("blocks", denoiser.blocks),
        ]
        var injected = 0
        for (name, blocks) in groups {
            for (i, block) in blocks.enumerated() {
                let p = "\(name).\(i)"
                let qkv = try wrap(block.attn.qkv, "\(p).attn.qkv"), out = try wrap(block.attn.out, "\(p).attn.out")
                block.attn.update(modules: ModuleChildren(values: ["qkv": .value(qkv), "out": .value(out)]))
                guard let up = block.mlp.mlp[0] as? Linear, let down = block.mlp.mlp[2] as? Linear else {
                    throw SperidError.weightMismatch("turbo LoRA: unexpected MLP layout in \(p)")
                }
                let layers: [NestedItem<String, Module>] = [
                    .value(try wrap(up, "\(p).mlp.mlp.0")), .value(block.mlp.mlp[1] as! Module), .value(try wrap(down, "\(p).mlp.mlp.2")),
                ]
                block.mlp.update(modules: ModuleChildren(values: ["mlp": .array(layers)]))
                injected += 4
            }
        }
        guard injected * 4 == w.count else {
            throw SperidError.weightMismatch("turbo LoRA: \(injected) layers for \(w.count) tensors")
        }
        return state
    }
}
