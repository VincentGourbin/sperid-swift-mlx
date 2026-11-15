// BiRefNet.swift - BiRefNet (Swin-L, multi-scale input, ASPP-deformable decoder) and its background remover
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import SperidCore

final class BiRefNetDecoder: Module {
    @ModuleInfo(key: "ipt_blk5") var ipt5: SimpleConvs
    @ModuleInfo(key: "ipt_blk4") var ipt4: SimpleConvs
    @ModuleInfo(key: "ipt_blk3") var ipt3: SimpleConvs
    @ModuleInfo(key: "ipt_blk2") var ipt2: SimpleConvs
    @ModuleInfo(key: "ipt_blk1") var ipt1: SimpleConvs
    @ModuleInfo(key: "decoder_block4") var dec4: BasicDecBlk
    @ModuleInfo(key: "decoder_block3") var dec3: BasicDecBlk
    @ModuleInfo(key: "decoder_block2") var dec2: BasicDecBlk
    @ModuleInfo(key: "decoder_block1") var dec1: BasicDecBlk
    @ModuleInfo(key: "conv_out1") var convOut1: [Conv2d]
    @ModuleInfo(key: "lateral_block4") var lat4: [String: Conv2d]
    @ModuleInfo(key: "lateral_block3") var lat3: [String: Conv2d]
    @ModuleInfo(key: "lateral_block2") var lat2: [String: Conv2d]
    @ModuleInfo(key: "gdt_convs_4") var gdt4: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_3") var gdt3: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_2") var gdt2: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_attn_4") var gdtAttn4: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_attn_3") var gdtAttn3: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_attn_2") var gdtAttn2: [UnaryLayer]

    override init() {
        let c = [3072, 1536, 768, 384]
        _ipt5.wrappedValue = SimpleConvs(1024 * 3, c[0] / 8); _ipt4.wrappedValue = SimpleConvs(256 * 3, c[0] / 8)
        _ipt3.wrappedValue = SimpleConvs(64 * 3, c[1] / 8); _ipt2.wrappedValue = SimpleConvs(16 * 3, c[2] / 8)
        _ipt1.wrappedValue = SimpleConvs(3, c[3] / 8)
        _dec4.wrappedValue = BasicDecBlk(c[0] + c[0] / 8, c[1]); _dec3.wrappedValue = BasicDecBlk(c[1] + c[0] / 8, c[2])
        _dec2.wrappedValue = BasicDecBlk(c[2] + c[1] / 8, c[3]); _dec1.wrappedValue = BasicDecBlk(c[3] + c[2] / 8, c[3] / 2)
        _convOut1.wrappedValue = [Conv2d(inputChannels: c[3] / 2 + c[3] / 8, outputChannels: 1, kernelSize: 1)]
        _lat4.wrappedValue = ["conv": Conv2d(inputChannels: c[1], outputChannels: c[1], kernelSize: 1)]
        _lat3.wrappedValue = ["conv": Conv2d(inputChannels: c[2], outputChannels: c[2], kernelSize: 1)]
        _lat2.wrappedValue = ["conv": Conv2d(inputChannels: c[3], outputChannels: c[3], kernelSize: 1)]
        _gdt4.wrappedValue = convSeq(c[1], 16, kernel: 3, bnRelu: true); _gdt3.wrappedValue = convSeq(c[2], 16, kernel: 3, bnRelu: true)
        _gdt2.wrappedValue = convSeq(c[3], 16, kernel: 3, bnRelu: true)
        _gdtAttn4.wrappedValue = convSeq(16, 1, kernel: 1, bnRelu: false); _gdtAttn3.wrappedValue = convSeq(16, 1, kernel: 1, bnRelu: false)
        _gdtAttn2.wrappedValue = convSeq(16, 1, kernel: 1, bnRelu: false)
        super.init()
    }

    static func up(_ x: MLXArray, like ref: MLXArray) -> MLXArray { Resample.bilinear(x, height: ref.dim(1), width: ref.dim(2)) }

    func callAsFunction(_ x: MLXArray, _ x1: MLXArray, _ x2: MLXArray, _ x3: MLXArray, _ x4: MLXArray) -> MLXArray {
        func patches(_ ref: MLXArray) -> MLXArray { image2patches(x, height: ref.dim(1), width: ref.dim(2)) }
        var p4 = dec4(concatenated([x4, ipt5(patches(x4))], axis: -1))
        p4 = p4 * sigmoid(runSequence(gdtAttn4, runSequence(gdt4, p4)))
        var p3 = Self.up(p4, like: x3) + lat4["conv"]!(x3)
        p3 = dec3(concatenated([p3, ipt4(patches(p3))], axis: -1))
        p3 = p3 * sigmoid(runSequence(gdtAttn3, runSequence(gdt3, p3)))
        var p2 = Self.up(p3, like: x2) + lat3["conv"]!(x2)
        p2 = dec2(concatenated([p2, ipt3(patches(p2))], axis: -1))
        p2 = p2 * sigmoid(runSequence(gdtAttn2, runSequence(gdt2, p2)))
        var p1 = Self.up(p2, like: x1) + lat2["conv"]!(x1)
        p1 = dec1(concatenated([p1, ipt2(patches(p1))], axis: -1))
        p1 = Resample.bilinear(p1, height: x.dim(1), width: x.dim(2))
        p1 = concatenated([p1, ipt1(patches(p1))], axis: -1)
        return convOut1[0](p1)
    }
}

/// BiRefNet with the Swin-L backbone, as shipped in `speridlabs/idol` (`birefnet.safetensors`, fp16).
public final class BiRefNet: Module {
    @ModuleInfo(key: "bb") var bb: SwinLarge
    @ModuleInfo(key: "squeeze_module") var squeeze: [BasicDecBlk]
    @ModuleInfo(key: "decoder") var decoder: BiRefNetDecoder
    public let computeDType: DType
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]
    public static let inputSize = 1024

    public init(computeDType: DType = .float16) {
        self.computeDType = computeDType
        _bb.wrappedValue = SwinLarge()
        _squeeze.wrappedValue = [BasicDecBlk(3072 + 384 + 768 + 1536, 3072)]
        _decoder.wrappedValue = BiRefNetDecoder()
        super.init()
    }

    public static func load(url: URL, computeDType: DType = .float16) throws -> BiRefNet {
        try IdolWeights.birefnetContract.verify(SafetensorsHeader(contentsOf: url))
        let model = BiRefNet(computeDType: computeDType)
        var selected: [String: MLXArray] = [:]
        for (key, value) in try loadArrays(url: url) where !key.hasSuffix("num_batches_tracked") {
            var v = value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
            v = key.hasSuffix("relative_position_index") ? v.asType(.int32) : v.asType(computeDType)
            eval(v)
            selected[key] = v
        }
        try model.update(parameters: ModuleParameters.unflattened(selected), verify: .all)
        return model
    }

    /// Encoder of `_forward_enc`: the backbone at full and half resolution, concatenated, plus the context
    /// stack for the squeeze block. x: normalised NHWC [1, 1024, 1024, 3].
    func encode(_ x: MLXArray) -> [MLXArray] {
        let full = bb(x)
        let half = bb(Resample.bilinear(x, height: x.dim(1) / 2, width: x.dim(2) / 2))
        var f = zip(full, half).map { concatenated([$0, BiRefNetDecoder.up($1, like: $0)], axis: -1) }
        f[3] = concatenated([BiRefNetDecoder.up(f[0], like: f[3]), BiRefNetDecoder.up(f[1], like: f[3]),
                             BiRefNetDecoder.up(f[2], like: f[3]), f[3]], axis: -1)
        f.forEach { eval($0) }
        return f
    }

    /// Alpha logits → sigmoid, [1, 1024, 1024, 1].
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = encode(x)
        let sq = squeeze[0](f[3]); eval(sq)
        return sigmoid(decoder(x, f[0], f[1], f[2], sq).asType(.float32))
    }

    /// `remove_background`: RGB → bilinear (align corners) to 1024², ImageNet normalisation, forward,
    /// alpha back to the image size (align corners), clamp, ×255, truncated to uint8.
    public func alpha(for image: RGBAImage) -> [UInt8] {
        let rgb = MLXArray(image.pixels, [image.height, image.width, 4])[0..., 0..., 0..<3].asType(.float32) / 255
        var t = Resample.bilinear(rgb.expandedDimensions(axis: 0), height: Self.inputSize, width: Self.inputSize)
        t = ((t - MLXArray(Self.mean)) / MLXArray(Self.std)).asType(computeDType)
        let a = Resample.bilinear(self(t), height: image.height, width: image.width)
        let u8 = (clip(a, min: 0, max: 1) * 255).asType(.uint8)
        return u8.reshaped(-1).asArray(UInt8.self)
    }
}

/// The reference's background remover (`IdolPipeline.background_remover`).
public final class BiRefNetBackgroundRemover: BackgroundRemover, @unchecked Sendable {
    let model: BiRefNet
    public init(model: BiRefNet) { self.model = model }
    public convenience init(modelsDirectory: URL) throws {
        self.init(model: try BiRefNet.load(url: IdolWeights.birefnet.localURL(in: modelsDirectory)))
        eval(model.parameters())
    }
    public func alpha(for image: RGBAImage) throws -> [UInt8] { model.alpha(for: image) }
}
