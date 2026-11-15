// StudioModel.swift - state of Idol Studio: settings like the Space, the pipeline, one generation at a time
// Copyright 2026 Vincent Gourbin

import AppKit
import Foundation
import IdolCore
import Observation
import SperidCore

/// The pipeline is not Sendable; it only ever runs on `worker`, one job at a time.
final class PipelineBox: @unchecked Sendable {
    var pipeline: IdolPipeline?
}

@Observable
@MainActor
final class StudioModel {
    static let gaussianChoices = [65_536, 131_072, 262_144, 524_288, 1_048_576]

    // settings (the Space's defaults)
    var mode: IdolCropMode = .synthetic
    var steps = 50
    var cfg = 3.0
    var rescaleT = 3.0
    var seed: UInt64 = 42
    var randomSeed = false
    var numGaussians = 262_144
    /// idol-turbo: 5 guidance-free steps (≈ 20× fewer evaluations than the 50-step base with CFG).
    var turbo = true

    var inputURL: URL?
    var inputImage: NSImage?
    var presets: [ExamplePreset] = []
    var thumbnails: [String: NSImage] = [:]

    var status = "Loading the models…"
    var stage: IdolStage?
    var fraction = 0.0
    var busy = false
    var ready = false
    var cloud: GaussianCloud?
    var cloudVersion = 0
    var lastOutputFolder: URL?
    var timings = ""

    let root = ModelsDirectory.resolve()
    private let box = PipelineBox()
    private let worker = DispatchQueue(label: "idol.pipeline", qos: .userInitiated)
    private var cancelFlag = false

    func start() {
        // `idol-studio path/to/file.ply` shows that file right away (also how the viewer is checked from a script)
        if let ply = CommandLine.arguments.dropFirst().first(where: { $0.hasSuffix(".ply") }) {
            openPLY(URL(fileURLWithPath: ply))
        }
        let box = box, root = root
        worker.async {
            do {
                let start = Date()
                box.pipeline = try IdolPipeline(modelsDirectory: root)
                let s = Date().timeIntervalSince(start)
                Task { @MainActor in self.ready = true; self.status = String(format: "Ready (models loaded in %.0f s)", s) }
            } catch {
                Task { @MainActor in self.status = "Could not load the models: \(error.localizedDescription)" }
            }
        }
        Task { await loadExamples() }
    }

    func loadExamples() async {
        do {
            presets = try await ExampleStore.load(root: root)
            for p in presets {
                if let url = try? await ExampleStore.image(p, root: root), let img = NSImage(contentsOf: url) {
                    thumbnails[p.id] = img
                }
            }
        } catch { status = "Examples unavailable: \(error.localizedDescription)" }
    }

    func open(_ url: URL) {
        inputURL = url
        inputImage = NSImage(contentsOf: url)
    }

    func pick(_ p: ExamplePreset) {
        Task {
            if let url = try? await ExampleStore.image(p, root: root) { open(url) }
            mode = IdolCropMode(rawValue: p.mode) ?? .synthetic
            steps = p.steps; cfg = p.cfg; rescaleT = p.rescaleT; seed = p.seed; randomSeed = false
            numGaussians = p.numGaussians
        }
    }

    func cancel() { cancelFlag = true }

    /// Shows any 3DGS .ply (Idol's, the Python SDK's, other tools').
    func openPLY(_ url: URL) {
        do {
            cloud = try PLYReader.read(url); cloudVersion += 1
            timings = "\(url.lastPathComponent) · \(cloud!.count) Gaussians"
        } catch { status = "Could not read \(url.lastPathComponent): \(error.localizedDescription)" }
    }

    /// Debug toggle for RealityKit's (undocumented) quaternion order: redraw with the other layout.
    func flipQuaternionOrder() {
        if #available(macOS 27.0, *) { SplatEntity.wFirst.toggle() }
        cloudVersion += 1
        status = "Quaternion order: \(quaternionOrderLabel)"
    }

    var quaternionOrderLabel: String {
        if #available(macOS 27.0, *) { return SplatEntity.wFirst ? "w, x, y, z" : "x, y, z, w" }
        return "-"
    }

    func generate() {
        guard let url = inputURL, ready, !busy else { return }
        if randomSeed { seed = UInt64.random(in: 0..<(1 << 31)) }
        let request = IdolRequest(mode: mode, seed: seed, steps: turbo ? 5 : steps, cfgStrength: turbo ? 0 : cfg,
                                  rescaleT: rescaleT, numGaussians: numGaussians)
        let useTurbo = turbo
        busy = true; cancelFlag = false; status = "Generating…"
        let box = box, root = root
        worker.async { [weak self] in
            do {
                let image = try RGBAImage(contentsOf: url)
                var noise = SeededNoise(seed: request.seed)
                let output = try box.pipeline!.generate(
                    image: image, request: request, noise: &noise, useTurbo: useTurbo,
                    isCancelled: { [weak self] in MainActor.assumeIsolatedOrFalse { self?.cancelFlag ?? false } },
                    progress: { p in Task { @MainActor in self?.stage = p.stage; self?.fraction = p.fraction } })
                let name = url.deletingPathExtension().lastPathComponent + "-seed\(request.seed)"
                let folder = root.appendingPathComponent("idol-studio-outputs")
                try output.write(to: folder, name: name)
                let t = output.timings
                let summary = String(format: "%d Gaussians · %d steps · encode %.1f s · denoise %.0f s · decode %.1f s",
                                     output.cloud.count, request.steps, t["encode"] ?? 0, t["denoise"] ?? 0, t["decode"] ?? 0)
                Task { @MainActor in
                    guard let self else { return }
                    self.cloud = output.cloud; self.cloudVersion += 1; self.lastOutputFolder = folder
                    self.timings = summary; self.status = "Done"; self.busy = false; self.stage = nil
                }
            } catch is CancellationError {
                Task { @MainActor in self?.status = "Cancelled"; self?.busy = false; self?.stage = nil }
            } catch {
                Task { @MainActor in self?.status = "Failed: \(error.localizedDescription)"; self?.busy = false; self?.stage = nil }
            }
        }
    }
}

extension MainActor {
    /// Reads main-actor state from the worker without blocking it: the flag is only advisory.
    nonisolated static func assumeIsolatedOrFalse(_ body: @MainActor () -> Bool) -> Bool {
        Thread.isMainThread ? MainActor.assumeIsolated(body) : DispatchQueue.main.sync { MainActor.assumeIsolated(body) }
    }
}
