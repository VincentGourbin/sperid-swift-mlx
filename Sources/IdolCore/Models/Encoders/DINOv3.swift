// DINOv3.swift - DINOv3 ViT-H+/16 (`transformers` DINOv3ViTModel) and Idol's `feature1`
// Copyright 2026 Vincent Gourbin
// Facts from transformers 3a134bc `modeling_dinov3_vit.py` and the checkpoint's config.json (plan/02.4).

import Foundation
import MLX
import MLXFast
import MLXNN

public struct DINOv3Config: Sendable {
    public var hidden = 1280, layers = 32, heads = 20, intermediate = 5120, patch = 16, registers = 4
    public var layerNormEps: Float = 1e-5
    public var ropeTheta: Double = 100
    public init() {}
}

final class DINOv3Attention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var o: Linear
    let heads: Int

    init(_ c: DINOv3Config) {
        heads = c.heads
        _q.wrappedValue = Linear(c.hidden, c.hidden)
        _k.wrappedValue = Linear(c.hidden, c.hidden, bias: false)  // key_bias: false
        _v.wrappedValue = Linear(c.hidden, c.hidden)
        _o.wrappedValue = Linear(c.hidden, c.hidden)
        super.init()
    }

    /// RoPE on the patch tokens only (the `prefix` CLS + registers are not rotated), `rotate_half` halves.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, prefix: Int) -> MLXArray {
        let (b, l, c) = (x.dim(0), x.dim(1), x.dim(2))
        let d = c / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped(b, l, heads, d).transposed(0, 2, 1, 3) }
        func rope(_ t: MLXArray) -> MLXArray {
            let head = t[0..., 0..., 0..<prefix], patches = t[0..., 0..., prefix...]
            let x1 = patches[.ellipsis, 0..<(d / 2)], x2 = patches[.ellipsis, (d / 2)...]
            let rotated = patches * cos + concatenated([-x2, x1], axis: -1) * sin
            return concatenated([head, rotated], axis: 2)
        }
        let out = MLXFast.scaledDotProductAttention(queries: rope(split(q(x))), keys: rope(split(k(x))),
                                                    values: split(v(x)), scale: 1 / Float(d).squareRoot(), mask: nil)
        return o(out.transposed(0, 2, 1, 3).reshaped(b, l, c))
    }
}

final class DINOv3GatedMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: DINOv3Config) {
        _gate.wrappedValue = Linear(c.hidden, c.intermediate)
        _up.wrappedValue = Linear(c.hidden, c.intermediate)
        _down.wrappedValue = Linear(c.intermediate, c.hidden)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class DINOv3LayerScale: Module {
    @ParameterInfo(key: "lambda1") var lambda1: MLXArray
    init(_ c: Int) { _lambda1.wrappedValue = MLXArray.ones([c]); super.init() }
}

final class DINOv3Layer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attention") var attention: DINOv3Attention
    @ModuleInfo(key: "layer_scale1") var scale1: DINOv3LayerScale
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: DINOv3GatedMLP
    @ModuleInfo(key: "layer_scale2") var scale2: DINOv3LayerScale

    init(_ c: DINOv3Config) {
        _norm1.wrappedValue = LayerNorm(dimensions: c.hidden, eps: c.layerNormEps)
        _attention.wrappedValue = DINOv3Attention(c)
        _scale1.wrappedValue = DINOv3LayerScale(c.hidden)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hidden, eps: c.layerNormEps)
        _mlp.wrappedValue = DINOv3GatedMLP(c)
        _scale2.wrappedValue = DINOv3LayerScale(c.hidden)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, prefix: Int) -> MLXArray {
        let h = x + attention(norm1(x), cos: cos, sin: sin, prefix: prefix) * scale1.lambda1
        return h + mlp(norm2(h)) * scale2.lambda1
    }
}

final class DINOv3Embeddings: Module {
    @ParameterInfo(key: "cls_token") var cls: MLXArray
    @ParameterInfo(key: "mask_token") var mask: MLXArray
    @ParameterInfo(key: "register_tokens") var registers: MLXArray
    @ModuleInfo(key: "patch_embeddings") var patch: Conv2d

    init(_ c: DINOv3Config) {
        _cls.wrappedValue = MLXArray.zeros([1, 1, c.hidden])
        _mask.wrappedValue = MLXArray.zeros([1, 1, c.hidden])
        _registers.wrappedValue = MLXArray.zeros([1, c.registers, c.hidden])
        _patch.wrappedValue = Conv2d(inputChannels: 3, outputChannels: c.hidden, kernelSize: .init(c.patch), stride: .init(c.patch))
        super.init()
    }
}

/// DINOv3 ViT-H+/16. `tokens(_:)` is Idol's `_dinov3_tokens`: ImageNet normalisation in fp32, the
/// model in `computeDType` (reference: bf16), `norm` of the model, then **another** parameter-free
/// layer_norm in fp32 (pitfall 11) → `feature1` [1, 5 + (H/16)², 1280] fp32.
public final class DINOv3: Module {
    @ModuleInfo(key: "embeddings") var embeddings: DINOv3Embeddings
    @ModuleInfo(key: "layer") var layers: [DINOv3Layer]
    @ModuleInfo(key: "norm") var norm: LayerNorm
    public let config: DINOv3Config
    public let computeDType: DType
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    public init(config: DINOv3Config = .init(), computeDType: DType = .bfloat16) {
        self.config = config
        self.computeDType = computeDType
        _embeddings.wrappedValue = DINOv3Embeddings(config)
        _layers.wrappedValue = (0..<config.layers).map { _ in DINOv3Layer(config) }
        _norm.wrappedValue = LayerNorm(dimensions: config.hidden, eps: config.layerNormEps)
        super.init()
    }

    /// `model.safetensors` of facebook/dinov3-vith16plus (615 tensors, F32): conv [O, I, kh, kw] → [O, kh, kw, I].
    public static func load(url: URL, computeDType: DType = .bfloat16) throws -> DINOv3 {
        let model = DINOv3(computeDType: computeDType)
        var selected: [String: MLXArray] = [:]
        for (key, value) in try loadArrays(url: url) {
            var v = value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
            v = v.asType(computeDType)
            eval(v)
            selected[key] = v
        }
        try model.update(parameters: ModuleParameters.unflattened(selected), verify: .all)
        return model
    }

    /// 2D RoPE of a `rows × cols` patch grid: centres (arange(0.5, n)/n)·2 − 1, 16 inverse frequencies
    /// 1/θ^(4k/64), angles 2π·coord·f (h then w) tiled twice → cos, sin [rows·cols, 64], in `dtype`
    /// (the reference casts them to the pixels' dtype).
    func rope(rows: Int, cols: Int, dtype: DType) -> (MLXArray, MLXArray) {
        let d = config.hidden / config.heads
        let inv = (0..<(d / 4)).map { k in Float(1 / pow(config.ropeTheta, Double(4 * k) / Double(d))) }
        var angles = [Float]()
        angles.reserveCapacity(rows * cols * d)
        let twoPi = 2 * piFloat32
        for r in 0..<rows {
            let y = (Float(r) + 0.5) / Float(rows) * 2 - 1
            for c in 0..<cols {
                let x = (Float(c) + 0.5) / Float(cols) * 2 - 1
                let row = inv.map { twoPi * y * $0 } + inv.map { twoPi * x * $0 }
                angles += row + row
            }
        }
        let a = MLXArray(angles, [rows * cols, d])
        return (cos(a).asType(dtype), sin(a).asType(dtype))
    }

    /// pixels [1, 3, H, W] ∈ [0, 1] → `feature1` (fp32).
    public func tokens(_ pixels: MLXArray) -> MLXArray {
        let mean = MLXArray(Self.mean).reshaped(1, 3, 1, 1), std = MLXArray(Self.std).reshaped(1, 3, 1, 1)
        let x = ((pixels.asType(.float32) - mean) / std).transposed(0, 2, 3, 1).asType(computeDType)
        let p = embeddings.patch(x)                                   // [1, H/16, W/16, C]
        let (b, rows, cols, c) = (p.dim(0), p.dim(1), p.dim(2), p.dim(3))
        var h = concatenated([embeddings.cls, embeddings.registers, p.reshaped(b, rows * cols, c)], axis: 1)
        let (cosA, sinA) = rope(rows: rows, cols: cols, dtype: computeDType)
        let prefix = 1 + config.registers
        for layer in layers { h = layer(h, cos: cosA, sin: sinA, prefix: prefix) }
        return plainLayerNorm(norm(h).asType(.float32))
    }
}
