// Norms.swift - LayerNorm32 and MultiHeadRMSNorm (`idol/models/layers.py`)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXFast
import MLXNN

/// `LayerNorm32`: LayerNorm computed in fp32, returned in the input dtype. Two eps live in the
/// reference (pitfall 18): 1e-6 inside transformer blocks, 1e-5 (torch default) everywhere else.
public final class LayerNorm32: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray?
    @ParameterInfo(key: "bias") var bias: MLXArray?
    let eps: Float

    public init(_ dimensions: Int, affine: Bool = true, eps: Float = 1e-5) {
        self.eps = eps
        _weight.wrappedValue = affine ? MLXArray.ones([dimensions]) : nil
        _bias.wrappedValue = affine ? MLXArray.zeros([dimensions]) : nil
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x.asType(.float32), weight: weight?.asType(.float32), bias: bias?.asType(.float32),
                          eps: eps).asType(x.dtype)
    }
}

/// `F.layer_norm(x, x.shape[-1:])` with no parameters (eps 1e-5), in fp32, input dtype out.
public func plainLayerNorm(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    MLXFast.layerNorm(x.asType(.float32), eps: eps).asType(x.dtype)
}

/// `MultiHeadRMSNorm`: `normalize(x.float(), dim=-1) · gamma · √D` — an L2 normalisation
/// (eps 1e-12 on the norm, `F.normalize`), not RMSNorm with eps (pitfall 4). `gamma` is [H, D].
public final class MultiHeadRMSNorm: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    let scale: Float

    public init(dim: Int, heads: Int) {
        scale = Float(dim).squareRoot()
        _gamma.wrappedValue = MLXArray.ones([heads, dim])
        super.init()
    }

    /// x: [..., H, D].
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let norm = maximum(sqrt(sum(xf * xf, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        return ((xf / norm) * gamma.asType(.float32) * scale).asType(x.dtype)
    }
}
