// UnifiedBlock.swift - the denoiser's attention and transformer block (`idol/models/layers.py`)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXNN

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
        let q = qNorm(applyInterleavedRotary(p[0..., 0..., 0], cos: cos, sin: sin))
        let k = kNorm(applyInterleavedRotary(p[0..., 0..., 1], cos: cos, sin: sin))
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
