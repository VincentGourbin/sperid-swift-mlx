// FluxVAEEncoder.swift - the encoder half of FLUX.2's `AutoencoderKLFlux2` (diffusers layout)
// Copyright 2026 Vincent Gourbin
// Structure after diffusers `Encoder` / `ResnetBlock2D` / `Downsample2D` / `Attention` (the code the
// Idol SDK runs); the decoder is never used by Idol. Activations are NHWC (MLX convolutions).

import Foundation
import MLX
import MLXNN

/// diffusers `ResnetBlock2D` without time embedding: norm → SiLU → conv → norm → SiLU → conv, + shortcut.
final class VAEResnetBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var shortcut: Conv2d?

    init(_ cin: Int, _ cout: Int) {
        _norm1.wrappedValue = GroupNorm(groupCount: 32, dimensions: cin, eps: 1e-6, pytorchCompatible: true)
        _conv1.wrappedValue = Conv2d(inputChannels: cin, outputChannels: cout, kernelSize: 3, padding: 1)
        _norm2.wrappedValue = GroupNorm(groupCount: 32, dimensions: cout, eps: 1e-6, pytorchCompatible: true)
        _conv2.wrappedValue = Conv2d(inputChannels: cout, outputChannels: cout, kernelSize: 3, padding: 1)
        _shortcut.wrappedValue = cin == cout ? nil : Conv2d(inputChannels: cin, outputChannels: cout, kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(norm1(x)))
        h = conv2(silu(norm2(h)))
        return (shortcut?(x) ?? x) + h
    }
}

/// diffusers `Downsample2D` with `padding = 0`: zero-pad right and bottom by one, then a stride-2 conv.
final class VAEDownsample: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(_ c: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 0)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(padded(x, widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))]))
    }
}

/// The mid-block attention: one head of width C over the H·W tokens, residual.
final class VAEAttention: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]

    init(_ c: Int) {
        _groupNorm.wrappedValue = GroupNorm(groupCount: 32, dimensions: c, eps: 1e-6, pytorchCompatible: true)
        _toQ.wrappedValue = Linear(c, c)
        _toK.wrappedValue = Linear(c, c)
        _toV.wrappedValue = Linear(c, c)
        _toOut.wrappedValue = [Linear(c, c)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let t = groupNorm(x).reshaped(b, h * w, c)
        let q = toQ(t).expandedDimensions(axis: 1), k = toK(t).expandedDimensions(axis: 1)
        let v = toV(t).expandedDimensions(axis: 1)
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                  scale: 1 / Float(c).squareRoot(), mask: nil)
        return x + toOut[0](o.squeezed(axis: 1)).reshaped(b, h, w, c)
    }
}

final class VAEDownBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [VAEResnetBlock]
    @ModuleInfo(key: "downsamplers") var downsamplers: [VAEDownsample]

    init(_ cin: Int, _ cout: Int, layers: Int, downsample: Bool) {
        _resnets.wrappedValue = (0..<layers).map { VAEResnetBlock($0 == 0 ? cin : cout, cout) }
        _downsamplers.wrappedValue = downsample ? [VAEDownsample(cout)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        for d in downsamplers { h = d(h) }
        return h
    }
}

final class VAEMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [VAEResnetBlock]
    @ModuleInfo(key: "attentions") var attentions: [VAEAttention]

    init(_ c: Int) {
        _resnets.wrappedValue = [VAEResnetBlock(c, c), VAEResnetBlock(c, c)]
        _attentions.wrappedValue = [VAEAttention(c)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { resnets[1](attentions[0](resnets[0](x))) }
}

final class VAEEncoderNet: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [VAEDownBlock]
    @ModuleInfo(key: "mid_block") var midBlock: VAEMidBlock
    @ModuleInfo(key: "conv_norm_out") var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(channels: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 2, latentChannels: Int = 32) {
        _convIn.wrappedValue = Conv2d(inputChannels: 3, outputChannels: channels[0], kernelSize: 3, padding: 1)
        _downBlocks.wrappedValue = channels.indices.map { i in
            VAEDownBlock(i == 0 ? channels[0] : channels[i - 1], channels[i], layers: layersPerBlock,
                         downsample: i < channels.count - 1)
        }
        _midBlock.wrappedValue = VAEMidBlock(channels.last!)
        _normOut.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels.last!, eps: 1e-6, pytorchCompatible: true)
        _convOut.wrappedValue = Conv2d(inputChannels: channels.last!, outputChannels: 2 * latentChannels,
                                       kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for block in downBlocks { h = block(h) }
        h = midBlock(h)
        return convOut(silu(normOut(h)))
    }
}

/// `bn.running_mean` / `bn.running_var` of the checkpoint (the latent BatchNorm statistics, 128 channels).
final class VAELatentStats: Module {
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray

    init(_ c: Int) {
        _runningMean.wrappedValue = MLXArray.zeros([c])
        _runningVar.wrappedValue = MLXArray.ones([c])
        super.init()
    }
}
