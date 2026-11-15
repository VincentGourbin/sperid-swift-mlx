// Denoiser.swift - `LatentSeqMMFlowModel` (`idol/models/dit.py`)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXNN

/// The rectified-flow denoiser: the object latent (8192 tokens) and the input camera (1 token),
/// conditioned on the image tokens (4101). Each stream is refined on its own (2 blocks), then the
/// 12 294 tokens go through 24 joint blocks with learned rotary positions. 560 tensors.
public final class IdolDenoiser: Module {
    @ModuleInfo(key: "t_embedder") var tEmbedder: SinusoidalEmbedder
    @ModuleInfo(key: "adaLN_modulation") var adaLN: [UnaryLayer]
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "cond_embedder") var condEmbedder: Linear
    @ModuleInfo(key: "cond_embedder2") var condEmbedder2: Linear
    @ParameterInfo(key: "pos_pe") var posPE: MLXArray
    @ModuleInfo(key: "noise_repo_layers") var noiseRePo: [RePo3DRotaryEmbedding]
    @ModuleInfo(key: "context_repo_layers") var contextRePo: [RePo3DRotaryEmbedding]
    @ModuleInfo(key: "repo_layers") var rePo: [RePo3DRotaryEmbedding]
    @ModuleInfo(key: "noise_refiner") var noiseRefiner: [UnifiedTransformerBlock]
    @ModuleInfo(key: "context_refiner") var contextRefiner: [UnifiedTransformerBlock]
    @ModuleInfo(key: "cam_refiner") var camRefiner: FeedForwardStack
    @ModuleInfo(key: "blocks") var blocks: [UnifiedTransformerBlock]
    @ParameterInfo(key: "shift_table") var shiftTable: MLXArray
    @ModuleInfo(key: "out_layer") var outLayer: Linear
    @ModuleInfo(key: "cam_out_layer") var camOutLayer: Linear
    let positionEmbedder: PcdPositionEmbedderV1
    public let blockDType: DType

    /// Prefixes of the modules the reference runs in fp16 (`convert_module_to_f16`).
    public static let halfPrecisionPrefixes = ["blocks.", "noise_refiner.", "context_refiner."]

    public init(qTokenLength: Int, inChannels: Int, camChannels: Int, modelChannels: Int, condChannels: Int,
                cond2Channels: Int, numBlocks: Int, numRefinerBlocks: Int, numHeads: Int, mlpRatio: Int = 4,
                blockDType: DType = .float16) {
        self.blockDType = blockDType
        let c = modelChannels
        positionEmbedder = PcdPositionEmbedderV1(channels: c)
        _tEmbedder.wrappedValue = SinusoidalEmbedder(hidden: c)
        _adaLN.wrappedValue = [SiLU(), Linear(c, 6 * c)]
        _inputLayer.wrappedValue = Linear(inChannels, c)
        _condEmbedder.wrappedValue = Linear(condChannels, c)
        _condEmbedder2.wrappedValue = Linear(cond2Channels, c)
        _posPE.wrappedValue = MLXArray.zeros([1, qTokenLength, 3])
        let rope = { (n: Int) in (0..<n).map { _ in RePo3DRotaryEmbedding(modelChannels: c, heads: numHeads) } }
        _noiseRePo.wrappedValue = rope(numRefinerBlocks)
        _contextRePo.wrappedValue = rope(numRefinerBlocks)
        _rePo.wrappedValue = rope(numBlocks)
        let block = { (n: Int, modulated: Bool) in
            (0..<n).map { _ in UnifiedTransformerBlock(channels: c, heads: numHeads, mlpRatio: mlpRatio, modulation: modulated) }
        }
        _noiseRefiner.wrappedValue = block(numRefinerBlocks, true)
        _contextRefiner.wrappedValue = block(numRefinerBlocks, false)
        _camRefiner.wrappedValue = FeedForwardStack(inChannels: camChannels, inner: c, out: c, layers: numRefinerBlocks)
        _blocks.wrappedValue = block(numBlocks, true)
        _shiftTable.wrappedValue = MLXArray.zeros([1, 2, c])
        _outLayer.wrappedValue = Linear(c, inChannels)
        _camOutLayer.wrappedValue = Linear(c, camChannels)
        super.init()
    }

    public convenience init(config: IdolConfig.Denoiser, blockDType: DType = .float16) {
        self.init(qTokenLength: config.qTokenLength, inChannels: config.inChannels, camChannels: config.camChannels,
                  modelChannels: config.modelChannels, condChannels: config.condChannels,
                  cond2Channels: config.cond2Channels, numBlocks: config.numBlocks,
                  numRefinerBlocks: config.numRefinerBlocks, numHeads: config.numHeads, mlpRatio: config.mlpRatio,
                  blockDType: blockDType)
    }

    /// Velocity of (latent [B, 8192, 16], camera [B, 1, 5]) at time `t` [B] ∈ [0, 1000], given the image
    /// tokens `feature1` [B, Lc, 1280] (DINOv3) and `feature2` [B, Lc, 128] (FLUX.2 VAE). fp32 outputs.
    public func callAsFunction(latent z: MLXArray, camera cam: MLXArray, t: MLXArray,
                               feature1: MLXArray, feature2: MLXArray) -> (latent: MLXArray, camera: MLXArray) {
        let bd = blockDType
        var hx = inputLayer(z).asType(bd)
        var hc = (condEmbedder(feature1) + condEmbedder2(feature2)).asType(bd)
        let tEmb = tEmbedder(t)
        var tMod = tEmb
        for layer in adaLN { tMod = layer(tMod) }
        tMod = tMod.asType(bd)

        hx = hx + positionEmbedder(posPE).asType(bd)
        for (rope, block) in zip(noiseRePo, noiseRefiner) {
            let (c, s) = rope(hx)
            hx = block(hx, mod: tMod, cos: c, sin: s)
        }
        for (rope, block) in zip(contextRePo, contextRefiner) {
            let (c, s) = rope(hc)
            hc = block(hc, mod: nil, cos: c, sin: s)
        }
        let hcam = camRefiner(cam).asType(bd)

        var h = concatenated([hx, hc, hcam], axis: 1)
        for (rope, block) in zip(rePo, blocks) {
            let (c, s) = rope(h)
            h = block(h, mod: tMod, cos: c, sin: s)
        }
        let lz = z.dim(1), total = h.dim(1)
        let parts = split(shiftTable + tEmb.expandedDimensions(axis: 1), parts: 2, axis: 1)
        let (shift, scale) = (parts[0], parts[1])
        let outX = plainLayerNorm(h[0..., 0..<lz].asType(.float32)) * (1 + scale) + shift
        let outCam = plainLayerNorm(h[0..., (total - cam.dim(1))...].asType(.float32)) * (1 + scale) + shift
        return (outLayer(outX), camOutLayer(outCam))
    }
}

/// `MLP` of `layers.py`: (layers − 1) × [Linear, GELU tanh] then Linear (keys `mlp.0`, `mlp.2`, …).
public final class FeedForwardStack: Module, UnaryLayer {
    @ModuleInfo(key: "mlp") var mlp: [UnaryLayer]

    public init(inChannels: Int, inner: Int, out: Int, layers: Int) {
        var stack: [UnaryLayer] = []
        for i in 0..<(layers - 1) {
            stack.append(Linear(i == 0 ? inChannels : inner, inner))
            stack.append(GELU(approximation: .tanh))
        }
        stack.append(Linear(layers == 1 ? inChannels : inner, out))
        _mlp.wrappedValue = stack
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { mlp.reduce(x) { $1($0) } }
}
