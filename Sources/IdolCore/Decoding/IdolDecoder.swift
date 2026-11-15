// IdolDecoder.swift - latent → anchors → Gaussians (`IdolPipeline.decode`)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import SperidCore

/// Both decoders, loaded together (≈ 0.6 GB in fp16). `decode` is the second half of the pipeline:
/// the octree places `numGaussians / 32` anchors, the Gaussian decoder turns each into 32 Gaussians.
public final class IdolDecoder {
    public let config: IdolConfig
    public let octree: OctreeDecoder
    public let gaussian: GaussianDecoder

    /// Loads `decoder.safetensors` and `decoder_gs.safetensors` from `<modelsDirectory>/speridlabs/idol`,
    /// checked against their header contracts first. `blockDType` .float16 is the reference precision.
    public init(modelsDirectory: URL, blockDType: DType = .float16) throws {
        config = try IdolConfig.load(from: IdolWeights.config.localURL(in: modelsDirectory))
        let precision = ModuleLoading.referencePrecision(blockDType: blockDType)
        octree = OctreeDecoder(config: config.decoder, blockDType: blockDType)
        gaussian = GaussianDecoder(config: config.decoderGS, blockDType: blockDType)
        for (file, contract, module) in [(IdolWeights.octreeDecoder, IdolWeights.octreeDecoderContract, octree as Module),
                                         (IdolWeights.gaussianDecoder, IdolWeights.gaussianDecoderContract, gaussian)] {
            let url = file.localURL(in: modelsDirectory)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw SperidError.missingFile("\(file.repoID)/\(file.path) (run `idol download`)")
            }
            try contract.verify(SafetensorsHeader(contentsOf: url))
            try ModuleLoading.load(module, from: loadArrays(url: url), dtype: precision)
        }
    }

    public struct Output {
        public let cloud: GaussianCloud
        public let anchors: MLXArray
        public let levels: [OctreeDescent.LevelStats]
    }

    /// latent [1, 8192, 16] → cloud of `anchorCount × 32` Gaussians.
    public func decode<N: NoiseSource>(latent: MLXArray, numGaussians: Int = 262_144, noise: inout N) -> Output {
        let perAnchor = config.decoderGS.representationConfig.numGaussians
        let numPoints = max(1, numGaussians / perAnchor)
        let descent = OctreeDescent.descend(decoder: octree, latent: latent, numPoints: numPoints,
                                            levels: config.maxVoxelLevel, noise: &noise)
        let anchors = MLXArray(descent.anchors, [1, numPoints, 3])
        let features = gaussian(anchors, latent: latent)
        let cloud = GaussianAssembly.assemble(anchors: anchors, features: features, decoder: gaussian)
        return Output(cloud: cloud, anchors: anchors, levels: descent.stats)
    }
}
