// Attention.swift - decoder attention (self / cross, RMS-normalised q and k) and the MLP
// Copyright 2026 Vincent Gourbin

import MLX
import MLXFast
import MLXNN

/// `FeedForwardNet`: Linear(C → C·ratio), GELU **tanh**, Linear (keys `mlp.0`, `mlp.2`, pitfall 5).
public final class FeedForwardNet: Module, UnaryLayer {
    @ModuleInfo(key: "mlp") var mlp: [UnaryLayer]

    public init(channels: Int, mlpRatio: Int = 4) {
        _mlp.wrappedValue = [Linear(channels, channels * mlpRatio), GELU(approximation: .tanh),
                             Linear(channels * mlpRatio, channels)]
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        mlp.reduce(x) { h, layer in layer(h) }
    }
}

/// [B, L, H, D] → [B, H, L, D] attention → [B, L, H·D]; no mask, scale 1/√D (torch SDPA default).
func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
    let (b, l, h, d) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
    let out = MLXFast.scaledDotProductAttention(
        queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
        scale: 1 / Float(d).squareRoot(), mask: nil)
    return out.transposed(0, 2, 1, 3).reshaped(b, l, h * d)
}

/// `MultiHeadAttention` of the decoders: self (`to_qkv`) or cross (`to_q`, `to_kv` from the latent).
public final class MultiHeadAttention: Module {
    let heads: Int
    let cross: Bool
    @ModuleInfo(key: "to_qkv") var toQKV: Linear?
    @ModuleInfo(key: "to_q") var toQ: Linear?
    @ModuleInfo(key: "to_kv") var toKV: Linear?
    @ModuleInfo(key: "q_rms_norm") var qNorm: MultiHeadRMSNorm
    @ModuleInfo(key: "k_rms_norm") var kNorm: MultiHeadRMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear

    public init(channels: Int, heads: Int, contextChannels: Int? = nil) {
        self.heads = heads
        cross = contextChannels != nil
        let d = channels / heads
        if let ctx = contextChannels {
            _toQ.wrappedValue = Linear(channels, channels)
            _toKV.wrappedValue = Linear(ctx, channels * 2)
        } else {
            _toQKV.wrappedValue = Linear(channels, channels * 3)
        }
        _qNorm.wrappedValue = MultiHeadRMSNorm(dim: d, heads: heads)
        _kNorm.wrappedValue = MultiHeadRMSNorm(dim: d, heads: heads)
        _toOut.wrappedValue = Linear(channels, channels)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, context: MLXArray? = nil) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let h: MLXArray
        if cross, let toQ, let toKV, let context {
            let q = qNorm(toQ(x).reshaped(b, l, heads, -1))
            let kv = toKV(context).reshaped(b, context.dim(1), 2, heads, -1)
            h = attend(q, kNorm(kv[0..., 0..., 0]), kv[0..., 0..., 1])
        } else {
            let qkv = toQKV!(x).reshaped(b, l, 3, heads, -1)
            h = attend(qNorm(qkv[0..., 0..., 0]), kNorm(qkv[0..., 0..., 1]), qkv[0..., 0..., 2])
        }
        return toOut(h)
    }
}
