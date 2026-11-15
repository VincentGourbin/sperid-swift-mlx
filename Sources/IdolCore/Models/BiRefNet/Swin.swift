// Swin.swift - the Swin-L backbone of BiRefNet (`idol/models/birefnet.py`, `_SwinLarge`)
// Copyright 2026 Vincent Gourbin
// Tokens are [1, H·W, C]; images NHWC. Window 12, shift 6 on odd blocks, relative position bias.

import Foundation
import MLX
import MLXFast
import MLXNN

final class SwinMlp: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(_ c: Int, _ hidden: Int) {
        _fc1.wrappedValue = Linear(c, hidden)
        _fc2.wrappedValue = Linear(hidden, c)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }  // nn.GELU(): exact erf
}

final class SwinWindowAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "relative_position_bias_table") var biasTable: MLXArray
    @ParameterInfo(key: "relative_position_index") var biasIndex: MLXArray
    let heads: Int, window: Int

    init(_ c: Int, heads: Int, window: Int) {
        self.heads = heads
        self.window = window
        _qkv.wrappedValue = Linear(c, 3 * c)
        _proj.wrappedValue = Linear(c, c)
        _biasTable.wrappedValue = MLXArray.zeros([(2 * window - 1) * (2 * window - 1), heads])
        _biasIndex.wrappedValue = MLXArray.zeros([window * window, window * window], dtype: .int32)
        super.init()
    }

    /// x: [nW, N, C] windows; mask: [nW, N, N] additive (shifted blocks) or nil.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (nw, n, c) = (x.dim(0), x.dim(1), x.dim(2))
        let d = c / heads
        let p = qkv(x).reshaped(nw, n, 3, heads, d).transposed(2, 0, 3, 1, 4)        // [3, nW, h, N, d]
        let q = p[0] * Float(pow(Double(d), -0.5)), k = p[1], v = p[2]
        var attn = matmul(q, k.transposed(0, 1, 3, 2))                                 // [nW, h, N, N]
        let bias = biasTable.take(biasIndex.reshaped(-1).asType(.int32), axis: 0).reshaped(n, n, heads).transposed(2, 0, 1)
        attn = attn + bias.expandedDimensions(axis: 0).asType(attn.dtype)
        if let mask { attn = attn + mask.expandedDimensions(axis: 1).asType(attn.dtype) }
        attn = softmax(attn.asType(.float32), axis: -1).asType(v.dtype)
        return proj(matmul(attn, v).transposed(0, 2, 1, 3).reshaped(nw, n, c))
    }
}

final class SwinBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: SwinWindowAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SwinMlp
    let window: Int, shift: Int

    init(_ c: Int, heads: Int, window: Int, shift: Int) {
        self.window = window
        self.shift = shift
        _norm1.wrappedValue = LayerNorm(dimensions: c)
        _attn.wrappedValue = SwinWindowAttention(c, heads: heads, window: window)
        _norm2.wrappedValue = LayerNorm(dimensions: c)
        _mlp.wrappedValue = SwinMlp(c, 4 * c)
        super.init()
    }

    static func partition(_ x: MLXArray, _ w: Int) -> MLXArray {                       // [1, Hp, Wp, C] → [nW, w·w, C]
        let (hp, wp, c) = (x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped(hp / w, w, wp / w, w, c).transposed(0, 2, 1, 3, 4).reshaped(-1, w * w, c)
    }

    static func reverse(_ x: MLXArray, _ w: Int, _ hp: Int, _ wp: Int) -> MLXArray {
        let c = x.dim(2)
        return x.reshaped(hp / w, wp / w, w, w, c).transposed(0, 2, 1, 3, 4).reshaped(1, hp, wp, c)
    }

    func callAsFunction(_ x: MLXArray, h: Int, w: Int, mask: MLXArray) -> MLXArray {
        let c = x.dim(2)
        var t = norm1(x).reshaped(1, h, w, c)
        let padR = (window - w % window) % window, padB = (window - h % window) % window
        if padR > 0 || padB > 0 { t = padded(t, widths: [.init((0, 0)), .init((0, padB)), .init((0, padR)), .init((0, 0))]) }
        let (hp, wp) = (h + padB, w + padR)
        if shift > 0 { t = roll(roll(t, shift: -shift, axis: 1), shift: -shift, axis: 2) }
        var y = attn(Self.partition(t, window), mask: shift > 0 ? mask : nil)
        y = Self.reverse(y, window, hp, wp)
        if shift > 0 { y = roll(roll(y, shift: shift, axis: 1), shift: shift, axis: 2) }
        if padR > 0 || padB > 0 { y = y[0..., 0..<h, 0..<w] }
        let out = x + y.reshaped(1, h * w, c)
        return out + mlp(norm2(out))
    }
}

final class SwinPatchMerging: Module {
    @ModuleInfo(key: "reduction") var reduction: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm
    init(_ c: Int) {
        _reduction.wrappedValue = Linear(4 * c, 2 * c, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: 4 * c)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, h: Int, w: Int) -> MLXArray {
        let c = x.dim(2)
        let g = x.reshaped(1, h, w, c)
        let x0 = g[0..., .stride(by: 2), .stride(by: 2)], x1 = g[0..., .stride(from: 1, by: 2), .stride(by: 2)]
        let x2 = g[0..., .stride(by: 2), .stride(from: 1, by: 2)], x3 = g[0..., .stride(from: 1, by: 2), .stride(from: 1, by: 2)]
        return reduction(norm(concatenated([x0, x1, x2, x3], axis: -1).reshaped(1, -1, 4 * c)))
    }
}

final class SwinLayer: Module {
    @ModuleInfo(key: "blocks") var blocks: [SwinBlock]
    @ModuleInfo(key: "downsample") var downsample: SwinPatchMerging?
    let window: Int

    init(_ c: Int, depth: Int, heads: Int, window: Int, downsample: Bool) {
        self.window = window
        _blocks.wrappedValue = (0..<depth).map { SwinBlock(c, heads: heads, window: window, shift: $0 % 2 == 0 ? 0 : window / 2) }
        _downsample.wrappedValue = downsample ? SwinPatchMerging(c) : nil
        super.init()
    }

    /// The shifted-window mask of `_SwinBasicLayer.forward`: regions labelled 0…8, −100 across regions.
    func mask(h: Int, w: Int) -> MLXArray {
        let hp = Int((Double(h) / Double(window)).rounded(.up)) * window, wp = Int((Double(w) / Double(window)).rounded(.up)) * window
        let s = window / 2
        func band(_ i: Int, _ n: Int) -> Int { i < n - window ? 0 : (i < n - s ? 1 : 2) }
        var labels = [Float](repeating: 0, count: hp * wp)
        for y in 0..<hp { for x in 0..<wp { labels[y * wp + x] = Float(band(y, hp) * 3 + band(x, wp)) } }
        let windows = Self_partition(MLXArray(labels, [1, hp, wp, 1]), window).reshaped(-1, window * window)
        let diff = windows.expandedDimensions(axis: 1) - windows.expandedDimensions(axis: 2)
        return which(diff .!= 0, MLXArray(Float(-100)), MLXArray(Float(0)))
    }

    private func Self_partition(_ x: MLXArray, _ w: Int) -> MLXArray { SwinBlock.partition(x, w) }

    func callAsFunction(_ x: MLXArray, h: Int, w: Int) -> (out: MLXArray, h: Int, w: Int, next: MLXArray, nh: Int, nw: Int) {
        let m = mask(h: h, w: w)
        var t = x
        for block in blocks { t = block(t, h: h, w: w, mask: m) }
        if let downsample { return (t, h, w, downsample(t, h: h, w: w), (h + 1) / 2, (w + 1) / 2) }
        return (t, h, w, t, h, w)
    }
}

final class SwinPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    init(_ c: Int) {
        _proj.wrappedValue = Conv2d(inputChannels: 3, outputChannels: c, kernelSize: 4, stride: 4)
        _norm.wrappedValue = LayerNorm(dimensions: c)
        super.init()
    }
}

/// Swin-L: embed 192, depths 2-2-18-2, heads 6-12-24-48, window 12. Returns the 4 normalised feature maps NHWC.
final class SwinLarge: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: SwinPatchEmbed
    @ModuleInfo(key: "layers") var layers: [SwinLayer]
    @ModuleInfo(key: "norm0") var norm0: LayerNorm
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    static let dims = [192, 384, 768, 1536]

    override init() {
        let depths = [2, 2, 18, 2], heads = [6, 12, 24, 48]
        _patchEmbed.wrappedValue = SwinPatchEmbed(192)
        _layers.wrappedValue = (0..<4).map { SwinLayer(Self.dims[$0], depth: depths[$0], heads: heads[$0], window: 12, downsample: $0 < 3) }
        _norm0.wrappedValue = LayerNorm(dimensions: 192)
        _norm1.wrappedValue = LayerNorm(dimensions: 384)
        _norm2.wrappedValue = LayerNorm(dimensions: 768)
        _norm3.wrappedValue = LayerNorm(dimensions: 1536)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        let e = patchEmbed.proj(x)                                    // [1, H/4, W/4, 192]
        var (h, w) = (e.dim(1), e.dim(2))
        var t = patchEmbed.norm(e.reshaped(1, h * w, 192))
        let norms = [norm0, norm1, norm2, norm3]
        var outs: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            let r = layer(t, h: h, w: w)
            outs.append(norms[i](r.out).reshaped(1, r.h, r.w, Self.dims[i]))
            eval(outs[i], r.next)
            (t, h, w) = (r.next, r.nh, r.nw)
        }
        return outs
    }
}
