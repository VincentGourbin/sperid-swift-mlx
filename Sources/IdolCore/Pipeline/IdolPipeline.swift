// IdolPipeline.swift - one image → 3D Gaussians, entirely in Swift (`idol/pipeline.py`, `IdolPipeline`)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import SperidCore

/// What a generation produces, like the SDK's `IdolOutput`.
public struct IdolOutput {
    public let cloud: GaussianCloud
    /// The 1024 × 1024 image the model saw (background removed, cropped, on black).
    public let preprocessed: RGBAImage
    public let camera: CameraPose
    public let timings: [String: Double]

    /// `<name>.ply`, `<name>_input.png`, `<name>_camera.json`, like `python -m idol.sample`.
    public func write(to folder: URL, name: String) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try PLYWriter.write(cloud, to: folder.appendingPathComponent("\(name).ply"))
        try preprocessed.writePNG(to: folder.appendingPathComponent("\(name)_input.png"), opaque: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(camera).write(to: folder.appendingPathComponent("\(name)_camera.json"))
    }
}

public enum IdolStage: String, Sendable {
    case preprocessing, encoding, denoising, decoding, done
}

public struct IdolProgress: Sendable {
    public let stage: IdolStage
    /// Within `denoising`: steps done / total; 1 for the others once finished.
    public let fraction: Double
}

public final class IdolPipeline {
    public let config: IdolConfig
    public let dinov3: DINOv3
    public let vae: FluxVAEEncoder
    public let denoiser: IdolDenoiser
    public let decoder: IdolDecoder
    public let remover: BackgroundRemover

    /// Loads every network (≈ 3.5 GB at the reference precision: fp16 blocks, bf16 encoders).
    public init(modelsDirectory: URL, remover: BackgroundRemover = VisionBackgroundRemover()) throws {
        config = try IdolConfig.load(from: IdolWeights.config.localURL(in: modelsDirectory))
        dinov3 = try DINOv3.load(url: EncoderWeights.dinov3(modelsDirectory: modelsDirectory))
        vae = try FluxVAEEncoder.load(url: EncoderWeights.flux2VAE(modelsDirectory: modelsDirectory))
        denoiser = try IdolDenoiser.load(modelsDirectory: modelsDirectory, config: config)
        decoder = try IdolDecoder(modelsDirectory: modelsDirectory)
        self.remover = remover
    }

    /// The image the encoders see (`IdolPipeline.preprocess`).
    public func preprocess(_ image: RGBAImage, request: IdolRequest) throws -> RGBAImage {
        try ImagePreprocessor.prepare(image, margin: request.effectiveCropMargin, size: config.imageSize, remover: remover)
    }

    /// Image tokens of a preprocessed image.
    public func condition<N: NoiseSource>(_ prepared: RGBAImage, noise: inout N) -> IdolCondition {
        let pixels = ImagePreprocessor.pixels(prepared)
        let latentSide = config.imageSize / 8
        let eps = noise.vaePosterior(shape: [1, FluxVAEEncoder.latentChannels, latentSide, latentSide])
        let f1 = dinov3.tokens(pixels)
        let f2 = vae.tokens(pixels, noise: eps)
        eval(f1, f2)
        return IdolCondition(feature1: f1, feature2: f2)
    }

    public func generate<N: NoiseSource>(image: RGBAImage, request: IdolRequest = .init(), noise: inout N,
                                         guidance: FlowSampler.Guidance = .sequential,
                                         isCancelled: () -> Bool = { false },
                                         progress: (IdolProgress) -> Void = { _ in }) throws -> IdolOutput {
        try request.validate()
        var timings: [String: Double] = [:]
        func timed<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
            let start = Date()
            defer { timings[name] = Date().timeIntervalSince(start) }
            return try body()
        }
        progress(.init(stage: .preprocessing, fraction: 0))
        let prepared = try timed("preprocess") { try preprocess(image, request: request) }
        if isCancelled() { throw CancellationError() }
        progress(.init(stage: .encoding, fraction: 0))
        let condition = timed("encode") { self.condition(prepared, noise: &noise) }
        let d = config.denoiser
        let latentNoise = noise.latentNoise(shape: [1, d.qTokenLength, d.inChannels])
        let cameraNoise = noise.cameraNoise(shape: [1, 1, d.camChannels])
        let schedule = FlowSchedule(steps: request.steps, rescaleT: request.rescaleT)
        progress(.init(stage: .denoising, fraction: 0))
        let sample = try timed("denoise") {
            try FlowSampler.sample(denoiser: denoiser, latentNoise: latentNoise, cameraNoise: cameraNoise,
                                   condition: condition, schedule: schedule, cfgStrength: request.cfgStrength,
                                   guidance: guidance, isCancelled: isCancelled,
                                   observe: { i, _, _ in progress(.init(stage: .denoising, fraction: Double(i + 1) / Double(schedule.steps))) })
        }
        if isCancelled() { throw CancellationError() }
        progress(.init(stage: .decoding, fraction: 0))
        let cloud = timed("decode") { decoder.decode(latent: sample.latent, numGaussians: request.numGaussians, noise: &noise).cloud }
        let camera = CameraPose(latent: sample.camera.reshaped(-1).asArray(Float.self))
        progress(.init(stage: .done, fraction: 1))
        return IdolOutput(cloud: cloud, preprocessed: prepared, camera: camera, timings: timings)
    }
}
