// UnifiedBlock.swift - the denoiser's attention and transformer block (`idol/models/layers.py`)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Rotation (interleaved pairs, fp32, rounded to the activations' dtype like the reference) then
/// MultiHeadRMSNorm, for q and k in one compiled graph: the unfused chain was 10 % of a joint block
/// (13.7 ms of 134 ms at 12 294 tokens, `idol profile denoiser-ops`, 2026-11-15).
/// Inputs: q, k [B, L, H, D]; cos, sin [B, L, H, D/2] (fp32); gamma_q, gamma_k [H, D].
let fusedRotaryNorm: @Sendable ([MLXArray]) -> [MLXArray] = compile { a in
    func rotNorm(_ x: MLXArray, _ gamma: MLXArray) -> MLXArray {
        let shape = x.shape
        let pairs = x.asType(.float32).reshaped(Array(shape.dropLast()) + [shape.last! / 2, 2])
        let x0 = pairs[.ellipsis, 0], x1 = pairs[.ellipsis, 1]
        let r = stacked([x0 * a[2] - x1 * a[3], x0 * a[3] + x1 * a[2]], axis: -1).reshaped(shape).asType(x.dtype)
        let rf = r.asType(.float32)
        let norm = maximum(sqrt(sum(rf * rf, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        return ((rf / norm) * gamma.asType(.float32) * Float(shape.last!).squareRoot()).asType(x.dtype)
    }
    return [rotNorm(a[0], a[4]), rotNorm(a[1], a[5])]
}

/// `IDOL_FUSED_ROPE=0` turns the fusion off (A/B measurements); on by default.
let useFusedRotaryNorm = ProcessInfo.processInfo.environment["IDOL_FUSED_ROPE"] != "0"

/// `RopeMultiHeadAttention`: qkv → RoPE on q and k **before** their RMS norms (pitfall 1) → SDPA → out.
public final class RopeMultiHeadAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "q_norm") var qNorm: MultiHeadRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MultiHeadRMSNorm
    @ModuleInfo(key: "out") var out: Linear
    let heads: Int
    let headDim: Int

    public init(channels: Int, heads: Int) {
        self.heads = heads
        headDim = channels / heads
        _qkv.wrappedValue = Linear(channels, channels * 3)
        _qNorm.wrappedValue = MultiHeadRMSNorm(dim: headDim, heads: heads)
        _kNorm.wrappedValue = MultiHeadRMSNorm(dim: headDim, heads: heads)
        _out.wrappedValue = Linear(channels, channels)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let p = qkv(x).reshaped(b, l, 3, heads, headDim)
        let q: MLXArray, k: MLXArray
        if useFusedRotaryNorm {
            let r = fusedRotaryNorm([p[0..., 0..., 0], p[0..., 0..., 1], cos, sin, qNorm.gamma, kNorm.gamma])
            (q, k) = (r[0], r[1])
        } else {
            q = qNorm(applyInterleavedRotary(p[0..., 0..., 0], cos: cos, sin: sin))
            k = kNorm(applyInterleavedRotary(p[0..., 0..., 1], cos: cos, sin: sin))
        }
        return out(attend(q, k, p[0..., 0..., 2]))
    }
}

/// `UnifiedTransformerBlock`: RoPE self attention + MLP. Modulated blocks take the shared adaLN output
/// plus their own `shift_table` and use LayerNorms without affine; `modulation: false` (the condition
/// refiner) uses affine LayerNorms and no modulation. eps 1e-6.
public final class UnifiedTransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm32
    @ModuleInfo(key: "norm2") var norm2: LayerNorm32
    @ModuleInfo(key: "attn") var attn: RopeMultiHeadAttention
    @ModuleInfo(key: "mlp") var mlp: FeedForwardNet
    @ParameterInfo(key: "shift_table") var shiftTable: MLXArray?
    let modulation: Bool

    public init(channels: Int, heads: Int, mlpRatio: Int = 4, modulation: Bool = true) {
        self.modulation = modulation
        _norm1.wrappedValue = LayerNorm32(channels, affine: !modulation, eps: 1e-6)
        _norm2.wrappedValue = LayerNorm32(channels, affine: !modulation, eps: 1e-6)
        _attn.wrappedValue = RopeMultiHeadAttention(channels: channels, heads: heads)
        _mlp.wrappedValue = FeedForwardNet(channels: channels, mlpRatio: mlpRatio)
        _shiftTable.wrappedValue = modulation ? MLXArray.zeros([1, 6 * channels]) : nil
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mod: MLXArray?, cos: MLXArray, sin: MLXArray) -> MLXArray {
        guard modulation, let mod, let shiftTable else {
            let x = x + attn(norm1(x), cos: cos, sin: sin)
            return x + mlp(norm2(x))
        }
        let m = modulationChunks(mod + shiftTable.asType(mod.dtype))
        var h = norm1(x) * (1 + m[1]) + m[0]
        var x = x + attn(h, cos: cos, sin: sin) * m[2]
        h = norm2(x) * (1 + m[4]) + m[3]
        x = x + mlp(h) * m[5]
        return x
    }
}
