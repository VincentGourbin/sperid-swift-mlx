// RePo.swift - learned 3D rotary positions (`RePo3DRotaryEmbedding`, `idol/models/layers.py`)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXNN

/// Each token's features predict a 3D offset per head, turned into RoPE angles: head_dim 64 is split
/// 20 + 20 + 24 between x, y, z (`freqs_0/1/2`, 10 + 10 + 12 learned frequencies). Computed in fp32 from
/// the block's own input, once per block. Returns (cos, sin), each [B, L, H, 32].
public final class RePo3DRotaryEmbedding: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm32
    @ModuleInfo(key: "gate_map") var gateMap: Linear
    @ModuleInfo(key: "content_map") var contentMap: Linear
    @ModuleInfo(key: "final_map") var finalMap: Linear
    @ParameterInfo(key: "freqs_0") var freqs0: MLXArray
    @ParameterInfo(key: "freqs_1") var freqs1: MLXArray
    @ParameterInfo(key: "freqs_2") var freqs2: MLXArray
    let heads: Int

    public init(modelChannels: Int, heads: Int, headDim: Int = 64, hiddenRatio: Double = 0.125) {
        self.heads = heads
        let hidden = Int(Double(modelChannels) * hiddenRatio)
        _norm.wrappedValue = LayerNorm32(modelChannels, affine: true, eps: 1e-5)
        _gateMap.wrappedValue = Linear(modelChannels, hidden, bias: false)
        _contentMap.wrappedValue = Linear(modelChannels, hidden, bias: false)
        _finalMap.wrappedValue = Linear(hidden, 3 * heads, bias: false)
        let d0 = 2 * (headDim / 6)
        _freqs0.wrappedValue = MLXArray.zeros([d0 / 2])
        _freqs1.wrappedValue = MLXArray.zeros([d0 / 2])
        _freqs2.wrappedValue = MLXArray.zeros([(headDim - 2 * d0) / 2])
        super.init()
    }

    /// `clamp_mul(x, f) = x·tanh(f) + x·(f − tanh(f))`: x·f up to rounding, written as the reference does.
    static func clampMul(_ x: MLXArray, _ f: MLXArray) -> MLXArray {
        let ft = tanh(f)
        return x * ft + x * (f - ft)
    }

    public func callAsFunction(_ x: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let h = norm(x.asType(.float32))
        let out = finalMap(silu(gateMap(h)) * contentMap(h))
        let (b, l) = (out.dim(0), out.dim(1))
        let delta = out.reshaped(b, l, heads, 3)
        let freqs = [freqs0, freqs1, freqs2].map { $0.asType(.float32) }
        let angles = (0..<3).map { a in
            Self.clampMul(delta[0..., 0..., 0..., a].expandedDimensions(axis: -1), freqs[a]) * piFloat32
        }
        let ang = concatenated(angles, axis: -1)
        return (cos(ang), sin(ang))
    }
}

/// `apply_rotary_emb` (real-valued form): rotates **interleaved pairs** (x[2i], x[2i+1]) — MLX's
/// `traditional: true` convention, the opposite of YuE2 (pitfall 2). Computed in fp32, cast back.
public func applyInterleavedRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let shape = x.shape
    let pairs = x.asType(.float32).reshaped(Array(shape.dropLast()) + [shape.last! / 2, 2])
    let x0 = pairs[.ellipsis, 0], x1 = pairs[.ellipsis, 1]
    let rotated = stacked([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis: -1)
    return rotated.reshaped(shape).asType(x.dtype)
}
