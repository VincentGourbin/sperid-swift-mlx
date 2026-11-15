// GaussianDecoder.swift - `GaussianFixedlenDecoder` (`idol/models/decoder.py`)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXNN

/// Turns each anchor point into 32 Gaussians: self attention over all anchors, cross attention to the
/// object latent, MLP (16 blocks, 328 tensors). Output [B, L, 480] = 32 Gaussians × 15 raw channels,
/// laid out **per attribute, then per Gaussian** (pitfall 12); `GaussianAssembly` turns it into a cloud.
public final class GaussianDecoder: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "blocks") var blocks: [TransformerCrossBlock]
    @ModuleInfo(key: "out_proj") var outProj: Linear
    /// Fixed per-Gaussian offset pattern [32, 3] and the base offset scale [], stored with the weights.
    @ParameterInfo(key: "points_offset_perturbation") var pointsOffsetPerturbation: MLXArray
    @ParameterInfo(key: "base_offset_scale") var baseOffsetScale: MLXArray
    let positionEmbedder: PcdPositionEmbedderV2
    public let blockDType: DType
    public let representation: IdolConfig.Representation

    public init(inChannels: Int, modelChannels: Int, condChannels: Int, numBlocks: Int, numHeads: Int,
                representation: IdolConfig.Representation, mlpRatio: Int = 4, blockDType: DType = .float16) {
        self.blockDType = blockDType
        self.representation = representation
        positionEmbedder = PcdPositionEmbedderV2(channels: modelChannels)
        let perAnchor = representation.numGaussians
        _inProj.wrappedValue = Linear(inChannels, modelChannels)
        _inputLayer.wrappedValue = Linear(modelChannels, modelChannels)
        _blocks.wrappedValue = (0..<numBlocks).map { _ in
            TransformerCrossBlock(channels: modelChannels, contextChannels: condChannels, heads: numHeads,
                                  mlpRatio: mlpRatio)
        }
        _outProj.wrappedValue = Linear(modelChannels, perAnchor * GaussianAssembly.channelsPerGaussian)
        _pointsOffsetPerturbation.wrappedValue = MLXArray.zeros([perAnchor, 3])
        _baseOffsetScale.wrappedValue = MLXArray(Float(0))
        super.init()
    }

    public convenience init(config: IdolConfig.GaussianDecoder, blockDType: DType = .float16) {
        self.init(inChannels: config.inChannels, modelChannels: config.modelChannels,
                  condChannels: config.condChannels, numBlocks: config.numBlocks, numHeads: config.numHeads,
                  representation: config.representationConfig, mlpRatio: config.mlpRatio, blockDType: blockDType)
    }

    /// points: anchors [B, L, 3]; latent [B, Lc, C] → raw outputs [B, L, 480] (fp32).
    public func callAsFunction(_ points: MLXArray, latent: MLXArray) -> MLXArray {
        let (b, l, c) = (points.dim(0), points.dim(1), points.dim(2))
        var h = inProj(points) + positionEmbedder(points.reshaped(-1, c)).reshaped(b, l, -1)
        h = inputLayer(h)
        var out = h.asType(blockDType)
        let context = latent.asType(blockDType)
        for block in blocks { out = block(out, context: context) }
        return outProj(plainLayerNorm(out.asType(.float32)))
    }
}
