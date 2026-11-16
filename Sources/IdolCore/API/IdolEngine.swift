// IdolEngine.swift - the public entry point: load once, generate many (async, cancellable)
// Copyright 2026 Vincent Gourbin

import CoreGraphics
import Foundation
import SperidCore

/// Input image of a generation. Real alpha skips background removal.
public enum IdolImage: Sendable {
    case url(URL)
    case cgImage(CGImage)
    case rgba(RGBAImage)

    func load() throws -> RGBAImage {
        switch self {
        case .url(let url): return try RGBAImage(contentsOf: url)
        case .cgImage(let image): return RGBAImage(cgImage: image)
        case .rgba(let image): return image
        }
    }
}

extension CGImage: @retroactive @unchecked Sendable {}

/// Idol on Apple Silicon: one image → 3D Gaussian splat.
///
/// ```swift
/// let engine = try await IdolEngine.load()                       // ≈ 4 GB of weights, once
/// let result = try await engine.generate(.url(photo), settings: .turbo) { print($0.stage) }
/// try result.writePLY(to: plyURL)
/// try result.writeUSDZ(to: usdzURL)
/// ```
///
/// Every MLX call runs on the engine's own thread (`MLXThread`); calls are serialised, one generation
/// at a time. Cancelling the calling task stops the generation before its next denoising step or between the two
/// forwards of a guided step: ≈ 0.2 s in turbo, ≈ 5-7 s (one forward) in `.quality`. Releasing the last reference frees the weights
/// and ends the thread, so an app can drop the engine between objects and load it again.
public final class IdolEngine: @unchecked Sendable {
    public struct Configuration: Sendable {
        public enum BackgroundRemoval: Sendable { case birefnet, vision }
        /// BiRefNet is the reference's cut-out; Apple Vision needs no weights but misses subjects in busy scenes.
        public var backgroundRemoval: BackgroundRemoval = .birefnet
        /// MLX buffer cache during a generation. `.bounded()` (2 GB) keeps a turbo generation near 11 GB of
        /// process memory instead of 34; `.unbounded` leaves the host's setting alone.
        public var memory: IdolMemoryPolicy = .bounded()
        public init(backgroundRemoval: BackgroundRemoval = .birefnet, memory: IdolMemoryPolicy = .bounded()) {
            self.backgroundRemoval = backgroundRemoval
            self.memory = memory
        }
    }

    private let thread: MLXThread
    private let pipeline: IdolPipeline
    private let configuration: Configuration
    /// The idol-turbo LoRAs were found (`settings.turbo` needs them).
    public let hasTurbo: Bool
    public let modelsDirectory: URL

    private init(thread: MLXThread, pipeline: IdolPipeline, configuration: Configuration, modelsDirectory: URL) {
        self.thread = thread
        self.pipeline = pipeline
        self.configuration = configuration
        self.hasTurbo = pipeline.turbo != nil
        self.modelsDirectory = modelsDirectory
    }

    /// Ends the engine's MLX thread. A generation in flight holds the engine, so none is queued here.
    deinit { thread.stop() }

    /// Loads every network from `modelsDirectory` (see `IdolModels` to check or download them).
    public static func load(modelsDirectory: URL = ModelsDirectory.resolve(),
                            configuration: Configuration = .init()) async throws -> IdolEngine {
        let thread = MLXThread()
        let pipeline = try await thread.run {
            try UncheckedBox(IdolPipeline(modelsDirectory: modelsDirectory,
                                          remover: configuration.backgroundRemoval == .vision ? VisionBackgroundRemover() : nil))
        }.value
        return IdolEngine(thread: thread, pipeline: pipeline, configuration: configuration, modelsDirectory: modelsDirectory)
    }

    /// One image → one object. `progress` is called from the engine's thread.
    public func generate(_ image: IdolImage, settings: IdolSettings = .turbo,
                         progress: (@Sendable (IdolProgress) -> Void)? = nil) async throws -> IdolResult {
        if settings.turbo && !hasTurbo { throw SperidError.missingFile("idol-turbo LoRA (IdolModels.download(includeTurbo: true))") }
        let seed = settings.seed ?? UInt64.random(in: 0..<(1 << 31))
        let request = settings.request(seed: seed)
        let flag = CancellationFlag()
        let pipeline = self.pipeline, memory = configuration.memory
        return try await withTaskCancellationHandler {
            try await thread.run {
                let input = try image.load()
                var noise = SeededNoise(seed: seed)
                let output = try pipeline.generate(image: input, request: request, noise: &noise, useTurbo: settings.turbo,
                                                   memory: memory, isCancelled: { flag.isCancelled },
                                                   progress: { progress?($0) })
                return IdolResult(cloud: output.cloud, preprocessed: output.preprocessed, camera: output.camera,
                                  settings: settings, seed: seed, timings: output.timings)
            }
        } onCancel: { flag.cancel() }
    }
}

/// Moves a non-Sendable value out of the MLX thread once it is fully built (weights already evaluated).
struct UncheckedBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}
