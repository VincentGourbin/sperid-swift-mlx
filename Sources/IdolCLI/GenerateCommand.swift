// GenerateCommand.swift - `idol generate`: an image → .ply, _input.png, _camera.json (all Swift)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import IdolCore
import MLXProfiler
import SperidCore

struct GenerateCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate", abstract: "One image → 3D Gaussian splat, like `python -m idol.sample`.")

    @Option(help: "Input image (PNG, JPEG, WebP, HEIC…). Real alpha skips background removal.") var image: String
    @Option(help: "synthetic (renders, crop 1.2) | real (photos, crop 1.7).") var mode = "synthetic"
    @Option(help: "Seed (Swift stream: not the same numbers as torch for the same seed).") var seed: UInt64 = 42
    @Option(help: "Sampling steps.") var steps = 50
    @Option(help: "Guidance strength.") var cfg = 3.0
    @Option(name: .customLong("rescale-t"), help: "Time shift.") var rescaleT = 3.0
    @Option(help: "Gaussians (32 768 … 1 048 576).") var numGaussians = 262_144
    @Option(help: "Guidance: sequential | batched.") var guidance = "sequential"
    @Flag(help: "idol-turbo: 5 guidance-free steps with the phase LoRAs (overrides --steps and --cfg).") var turbo = false
    @Option(help: "Output folder.") var out = "outputs"
    @Option(help: "Models directory.") var modelsDir: String?
    @Option(help: "Profile with swift-mlx-profiler (fineGrained): writes report.txt and trace.json into this folder.")
    var profile: String?

    func run() throws {
        let root = modelsDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? ModelsDirectory.resolve()
        guard let cropMode = IdolCropMode(rawValue: mode) else { throw ValidationError("--mode synthetic|real") }
        let request = IdolRequest(mode: cropMode, seed: seed, steps: turbo ? 5 : steps, cfgStrength: turbo ? 0 : cfg,
                                  rescaleT: rescaleT, numGaussians: numGaussians)
        let session = profile.map { _ in ProfilingSession(config: .fineGrained) }
        if let session {
            session.title = "idol generate"
            session.metadata = ["image": image, "turbo": "\(turbo)", "steps": "\(request.steps)", "gaussians": "\(numGaussians)"]
            MLXProfiler.shared.enable()
            MLXProfiler.shared.activeSession = session
            session.beginPhase("1. Load models", category: .modelLoad)
        }
        let loadStart = Date()
        let pipeline = try IdolPipeline(modelsDirectory: root)
        session?.endPhase("1. Load models", category: .modelLoad)
        print(String(format: "loaded in %.1f s", Date().timeIntervalSince(loadStart)))
        let input = try RGBAImage(contentsOf: URL(fileURLWithPath: image))
        var noise = SeededNoise(seed: seed)
        var lastStage = ""
        let output = try pipeline.generate(image: input, request: request, noise: &noise,
                                           guidance: guidance == "batched" ? .batched : .sequential, useTurbo: turbo,
                                           progress: { p in
            if p.stage.rawValue != lastStage { print("  \(p.stage.rawValue)…"); lastStage = p.stage.rawValue }
        })
        let name = URL(fileURLWithPath: image).deletingPathExtension().lastPathComponent
        session?.beginPhase("8. Write .ply / png / json", category: .postProcess)
        try output.write(to: URL(fileURLWithPath: out), name: name)
        session?.endPhase("8. Write .ply / png / json", category: .postProcess)
        if let session, let profile {
            session.finish()
            let report = session.generateReport()
            print(report)
            let dir = URL(fileURLWithPath: profile)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try report.write(to: dir.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)
            try ChromeTraceExporter.export(session: session).write(to: dir.appendingPathComponent("trace.json"))
        }
        let t = output.timings
        print(String(format: "GENERATE OK %@: %d Gaussians — preprocess %.1f s, encode %.1f s, denoise %.1f s, decode %.1f s → %@/%@.ply",
                     name, output.cloud.count, t["preprocess"] ?? 0, t["encode"] ?? 0, t["denoise"] ?? 0, t["decode"] ?? 0, out, name))
    }
}
