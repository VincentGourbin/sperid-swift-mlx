// DenoiserLoading.swift - load the published denoiser at the reference precision
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import SperidCore

extension IdolDenoiser {
    /// `<modelsDirectory>/speridlabs/idol/denoiser.safetensors`, checked against its header contract,
    /// Linear layers of the blocks and refiners in `blockDType`, everything else fp32 (`plan/02.7`).
    public static func load(modelsDirectory: URL, config: IdolConfig, blockDType: DType = .float16) throws -> IdolDenoiser {
        let url = IdolWeights.denoiser.localURL(in: modelsDirectory)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SperidError.missingFile("\(IdolWeights.denoiser.repoID)/\(IdolWeights.denoiser.path) (run `idol download`)")
        }
        try IdolWeights.denoiserContract.verify(SafetensorsHeader(contentsOf: url))
        let model = IdolDenoiser(config: config.denoiser, blockDType: blockDType)
        try ModuleLoading.load(model, from: loadArrays(url: url),
                               dtype: ModuleLoading.referencePrecision(blockDType: blockDType, prefixes: halfPrecisionPrefixes))
        return model
    }
}
