// StudioModel.swift - state of Idol Studio: settings like the Space, the pipeline, one generation at a time
// Copyright 2026 Vincent Gourbin

import AppKit
import Foundation
import IdolCore
import Observation
import IdolUI
import SperidCore

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
    /// The public API (`IdolEngine`): its own MLX thread, async generation, cancellation by task.
    private var engine: IdolEngine?
    private var task: Task<Void, Never>?

    func start() {
        // `idol-studio path/to/file.ply` shows that file right away (also how the viewer is checked from a script)
        if let ply = CommandLine.arguments.dropFirst().first(where: { $0.hasSuffix(".ply") }) {
            openPLY(URL(fileURLWithPath: ply))
        }
        Task {
            do {
                let start = Date()
                let missing = IdolModels.status(modelsDirectory: root)
                if !missing.isComplete {
                    status = String(format: "Downloading %.1f GB of weights…", Double(missing.missingBytes) / 1e9)
                    try await IdolModels.download(modelsDirectory: root)
                }
                engine = try await IdolEngine.load(modelsDirectory: root)
                ready = true
                status = String(format: "Ready (models loaded in %.0f s)", Date().timeIntervalSince(start))
            } catch {
                status = "Could not load the models: \(error.localizedDescription)"
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

    func cancel() { task?.cancel() }

    /// Extracts the mesh of the current splat and writes it as .glb or .usdz (chosen by the extension).
    func exportMesh(to url: URL) {
        guard let cloud else { return }
        status = "Exporting \(url.lastPathComponent)…"
        let snapshot = cloud
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let mesh = MeshExtraction.extract(snapshot)
                if url.pathExtension.lowercased() == "glb" { try GLBWriter.write(mesh, to: url) } else { try USDZWriter.write(mesh, to: url) }
                Task { @MainActor in self.status = "Exported \(url.lastPathComponent) (\(mesh.triangleCount) triangles)" }
            } catch {
                Task { @MainActor in self.status = "Export failed: \(error.localizedDescription)" }
            }
        }
    }

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
        guard let url = inputURL, let engine, !busy else { return }
        let settings = IdolSettings(mode: mode, seed: randomSeed ? nil : seed, turbo: turbo, steps: steps,
                                    guidance: cfg, timeShift: rescaleT, gaussians: numGaussians)
        busy = true
        status = "Generating…"
        let root = root
        task = Task {
            do {
                let result = try await engine.generate(.url(url), settings: settings) { p in
                    Task { @MainActor in self.stage = p.stage; self.fraction = p.fraction }
                }
                seed = result.seed
                let folder = root.appendingPathComponent("idol-studio-outputs")
                try result.write(to: folder, name: url.deletingPathExtension().lastPathComponent + "-seed\(result.seed)")
                let t = result.timings
                timings = String(format: "%d Gaussians · %@ · encode %.1f s · denoise %.0f s · decode %.1f s",
                                 result.cloud.count, settings.turbo ? "turbo 5 steps" : "\(settings.steps) steps",
                                 t["encode"] ?? 0, t["denoise"] ?? 0, t["decode"] ?? 0)
                cloud = result.cloud; cloudVersion += 1; lastOutputFolder = folder; status = "Done"
            } catch is CancellationError {
                status = "Cancelled"
            } catch {
                status = "Failed: \(error.localizedDescription)"
            }
            busy = false; stage = nil
        }
    }
}
