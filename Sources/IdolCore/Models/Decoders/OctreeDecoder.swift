// OctreeDecoder.swift - `OctreeProbabilityFixedlenDecoder` (`idol/models/decoder.py`)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXNN

/// Predicts, for each octree cell, how its points split among its 8 children. Cross attention only
/// (to the object latent), modulated by the octree resolution. 4 blocks, 60 tensors.
public final class OctreeDecoder: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "l_embedder") var levelEmbedder: SinusoidalEmbedder
    @ModuleInfo(key: "adaLN_modulation") var adaLN: [UnaryLayer]
    @ModuleInfo(key: "blocks") var blocks: [ModulatedCrossOnlyBlock]
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let positionEmbedder: PcdPositionEmbedderV2
    /// The reference runs the blocks in fp16 (`self.dtype`); tiny fixtures use fp32.
    public let blockDType: DType

    public init(modelChannels: Int, condChannels: Int, numBlocks: Int, numHeads: Int, mlpRatio: Int = 4,
                blockDType: DType = .float16) {
        self.blockDType = blockDType
        positionEmbedder = PcdPositionEmbedderV2(channels: modelChannels)
        _inProj.wrappedValue = Linear(3, modelChannels)
        _inputLayer.wrappedValue = Linear(modelChannels, modelChannels)
        _levelEmbedder.wrappedValue = SinusoidalEmbedder(hidden: modelChannels, maxPeriod: 1024, twoPi: true)
        _adaLN.wrappedValue = [SiLU(), Linear(modelChannels, 6 * modelChannels)]
        _blocks.wrappedValue = (0..<numBlocks).map { _ in
            ModulatedCrossOnlyBlock(channels: modelChannels, contextChannels: condChannels, heads: numHeads,
                                    mlpRatio: mlpRatio)
        }
        _outProj.wrappedValue = Linear(modelChannels, 8)
        super.init()
    }

    public convenience init(config: IdolConfig.OctreeDecoder, blockDType: DType = .float16) {
        self.init(modelChannels: config.modelChannels, condChannels: config.condChannels,
                  numBlocks: config.numBlocks, numHeads: config.numHeads, mlpRatio: config.mlpRatio,
                  blockDType: blockDType)
    }

    /// x: cell centres [B, L, 3] ∈ [0, 1]; resolution: [B] (the cells' octree resolution 1 << l,
    /// pitfall 6); latent: [B, Lc, C]. Returns logits [B, L, 8] (fp32).
    public func callAsFunction(_ x: MLXArray, resolution: MLXArray, latent: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var h = inProj(x) + positionEmbedder(x.reshaped(-1, 3)).reshaped(b, l, -1)
        h = inputLayer(h)
        var mod = levelEmbedder(resolution.asType(h.dtype))
        for layer in adaLN { mod = layer(mod) }
        mod = mod.asType(blockDType)
        var out = h.asType(blockDType)
        let context = latent.asType(blockDType)
        for block in blocks { out = block(out, mod: mod, context: context) }
        return outProj(plainLayerNorm(out.asType(.float32)))
    }
}
