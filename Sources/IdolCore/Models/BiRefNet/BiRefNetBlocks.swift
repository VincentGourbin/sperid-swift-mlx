// BiRefNetBlocks.swift - ASPP-deformable and decoder blocks of BiRefNet (`idol/models/birefnet.py`)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXNN

/// Eval-mode BatchNorm2d (running statistics, eps 1e-5), NHWC, computed in fp32.
final class BN: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    @ParameterInfo(key: "running_mean") var mean: MLXArray
    @ParameterInfo(key: "running_var") var variance: MLXArray
    init(_ c: Int) {
        _weight.wrappedValue = MLXArray.ones([c]); _bias.wrappedValue = MLXArray.zeros([c])
        _mean.wrappedValue = MLXArray.zeros([c]); _variance.wrappedValue = MLXArray.ones([c])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scale = weight.asType(.float32) / sqrt(variance.asType(.float32) + 1e-5)
        return ((x.asType(.float32) - mean.asType(.float32)) * scale + bias.asType(.float32)).asType(x.dtype)
    }
}

final class DeformableConv2d: Module {
    @ModuleInfo(key: "offset_conv") var offsetConv: Conv2d
    @ModuleInfo(key: "modulator_conv") var modulatorConv: Conv2d
    @ModuleInfo(key: "regular_conv") var regularConv: Conv2d
    let k: Int, padding: Int
    init(_ cin: Int, _ cout: Int, kernel k: Int, padding: Int) {
        self.k = k
        self.padding = padding
        _offsetConv.wrappedValue = Conv2d(inputChannels: cin, outputChannels: 2 * k * k, kernelSize: .init(k), padding: .init(padding))
        _modulatorConv.wrappedValue = Conv2d(inputChannels: cin, outputChannels: k * k, kernelSize: .init(k), padding: .init(padding))
        _regularConv.wrappedValue = Conv2d(inputChannels: cin, outputChannels: cout, kernelSize: .init(k), padding: .init(padding), bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let offset = offsetConv(x)
        let mask = 2 * sigmoid(modulatorConv(x).asType(.float32))
        return DeformConv.apply(x, offset: offset, mask: mask, weight: regularConv.weight, bias: nil, kernel: k, padding: padding)
    }
}

final class ASPPModuleDeformable: Module {
    @ModuleInfo(key: "atrous_conv") var conv: DeformableConv2d
    @ModuleInfo(key: "bn") var bn: BN
    init(_ cin: Int, _ planes: Int, kernel: Int, padding: Int) {
        _conv.wrappedValue = DeformableConv2d(cin, planes, kernel: kernel, padding: padding)
        _bn.wrappedValue = BN(planes)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { relu(bn(conv(x))) }
}

final class ASPPDeformable: Module {
    @ModuleInfo(key: "aspp1") var aspp1: ASPPModuleDeformable
    @ModuleInfo(key: "aspp_deforms") var deforms: [ASPPModuleDeformable]
    @ModuleInfo(key: "global_avg_pool") var pool: [UnaryLayer]      // [AdaptiveAvgPool (no params), Conv, BN, ReLU]
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BN
    init(_ cin: Int) {
        let inter = 256
        _aspp1.wrappedValue = ASPPModuleDeformable(cin, inter, kernel: 1, padding: 0)
        _deforms.wrappedValue = [1, 3, 7].map { ASPPModuleDeformable(cin, inter, kernel: $0, padding: $0 / 2) }
        _pool.wrappedValue = [Identity(), Conv2d(inputChannels: cin, outputChannels: inter, kernelSize: 1, bias: false), BN(inter), ReLU()]
        _conv1.wrappedValue = Conv2d(inputChannels: inter * 5, outputChannels: cin, kernelSize: 1, bias: false)
        _bn1.wrappedValue = BN(cin)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x1 = aspp1(x)
        let d = deforms.map { $0(x) }
        var g = mean(x.asType(.float32), axes: [1, 2], keepDims: true).asType(x.dtype)   // AdaptiveAvgPool2d(1)
        for layer in pool.dropFirst() { g = layer(g) }
        let x5 = broadcast(g, to: [1, x1.dim(1), x1.dim(2), g.dim(3)])                   // interpolate 1×1 → H×W
        return relu(bn1(conv1(concatenated([x1] + d + [x5], axis: -1))))
    }
}

final class BasicDecBlk: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "bn_in") var bnIn: BN
    @ModuleInfo(key: "dec_att") var decAtt: ASPPDeformable
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    @ModuleInfo(key: "bn_out") var bnOut: BN
    init(_ cin: Int, _ cout: Int, inter: Int = 64) {
        _convIn.wrappedValue = Conv2d(inputChannels: cin, outputChannels: inter, kernelSize: 3, padding: 1)
        _bnIn.wrappedValue = BN(inter)
        _decAtt.wrappedValue = ASPPDeformable(inter)
        _convOut.wrappedValue = Conv2d(inputChannels: inter, outputChannels: cout, kernelSize: 3, padding: 1)
        _bnOut.wrappedValue = BN(cout)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { bnOut(convOut(decAtt(relu(bnIn(convIn(x)))))) }
}

final class SimpleConvs: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    init(_ cin: Int, _ cout: Int, inter: Int = 64) {
        _conv1.wrappedValue = Conv2d(inputChannels: cin, outputChannels: inter, kernelSize: 3, padding: 1)
        _convOut.wrappedValue = Conv2d(inputChannels: inter, outputChannels: cout, kernelSize: 3, padding: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { convOut(conv1(x)) }
}

/// `nn.Sequential(Conv2d, BatchNorm2d, ReLU)` (or the conv alone): numeric keys load as an array.
func convSeq(_ cin: Int, _ cout: Int, kernel: Int, bnRelu: Bool) -> [UnaryLayer] {
    let conv = Conv2d(inputChannels: cin, outputChannels: cout, kernelSize: .init(kernel), padding: .init(kernel / 2))
    return bnRelu ? [conv, BN(cout), ReLU()] : [conv]
}

func runSequence(_ layers: [UnaryLayer], _ x: MLXArray) -> MLXArray { layers.reduce(x) { $1($0) } }

/// `einops 'b c (hg h) (wg w) -> b (c hg wg) h w'` in NHWC: [1, H, W, C] → [1, h, w, C·hg·wg].
func image2patches(_ x: MLXArray, height h: Int, width w: Int) -> MLXArray {
    let (hf, wf, c) = (x.dim(1), x.dim(2), x.dim(3))
    let hg = hf / h, wg = wf / w
    return x.reshaped(1, hg, h, wg, w, c).transposed(0, 2, 4, 5, 1, 3).reshaped(1, h, w, c * hg * wg)
}
