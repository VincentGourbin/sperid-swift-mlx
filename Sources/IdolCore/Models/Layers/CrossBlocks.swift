// CrossBlocks.swift - the two decoder transformer blocks (`idol/models/layers.py`)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXNN

/// Splits [B, 6C] into six [B, 1, C] (shift/scale/gate for attention, then for the MLP).
func modulationChunks(_ mod: MLXArray) -> [MLXArray] {
    split(mod, parts: 6, axis: 1).map { $0.expandedDimensions(axis: 1) }
}

/// `ModulatedTransformerCrossOnlyBlock` (octree decoder): cross attention + MLP, adaLN-modulated by
/// the octree level; no self attention, no per-block shift table. Norms without affine, eps 1e-6.
public final class ModulatedCrossOnlyBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm32
    @ModuleInfo(key: "norm2") var norm2: LayerNorm32
    @ModuleInfo(key: "cross_attn") var crossAttn: MultiHeadAttention
    @ModuleInfo(key: "mlp") var mlp: FeedForwardNet

    public init(channels: Int, contextChannels: Int, heads: Int, mlpRatio: Int = 4) {
        _norm1.wrappedValue = LayerNorm32(channels, affine: false, eps: 1e-6)
        _norm2.wrappedValue = LayerNorm32(channels, affine: false, eps: 1e-6)
        _crossAttn.wrappedValue = MultiHeadAttention(channels: channels, heads: heads, contextChannels: contextChannels)
        _mlp.wrappedValue = FeedForwardNet(channels: channels, mlpRatio: mlpRatio)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mod: MLXArray, context: MLXArray) -> MLXArray {
        let m = modulationChunks(mod)
        var h = norm1(x) * (1 + m[1]) + m[0]
        var x = x + crossAttn(h, context: context) * m[2]
        h = norm2(x) * (1 + m[4]) + m[3]
        x = x + mlp(h) * m[5]
        return x
    }
}

/// `TransformerCrossBlock` (Gaussian decoder): self attention, cross attention to the latent, MLP.
/// norm1 and norm3 without affine, norm2 **with** affine; all eps 1e-6.
public final class TransformerCrossBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm32
    @ModuleInfo(key: "norm2") var norm2: LayerNorm32
    @ModuleInfo(key: "norm3") var norm3: LayerNorm32
    @ModuleInfo(key: "self_attn") var selfAttn: MultiHeadAttention
    @ModuleInfo(key: "cross_attn") var crossAttn: MultiHeadAttention
    @ModuleInfo(key: "mlp") var mlp: FeedForwardNet

    public init(channels: Int, contextChannels: Int, heads: Int, mlpRatio: Int = 4) {
        _norm1.wrappedValue = LayerNorm32(channels, affine: false, eps: 1e-6)
        _norm2.wrappedValue = LayerNorm32(channels, affine: true, eps: 1e-6)
        _norm3.wrappedValue = LayerNorm32(channels, affine: false, eps: 1e-6)
        _selfAttn.wrappedValue = MultiHeadAttention(channels: channels, heads: heads)
        _crossAttn.wrappedValue = MultiHeadAttention(channels: channels, heads: heads, contextChannels: contextChannels)
        _mlp.wrappedValue = FeedForwardNet(channels: channels, mlpRatio: mlpRatio)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        var x = x + selfAttn(norm1(x))
        x = x + crossAttn(norm2(x), context: context)
        return x + mlp(norm3(x))
    }
}
