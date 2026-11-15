// GenerateCommand.swift - `idol generate`: an image → .ply, _input.png, _camera.json (all Swift)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import IdolCore
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
    @Option(help: "Output folder.") var out = "outputs"
    @Option(help: "Models directory.") var modelsDir: String?

    func run() throws {
        let root = modelsDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? ModelsDirectory.resolve()
        guard let cropMode = IdolCropMode(rawValue: mode) else { throw ValidationError("--mode synthetic|real") }
        let request = IdolRequest(mode: cropMode, seed: seed, steps: steps, cfgStrength: cfg, rescaleT: rescaleT,
                                  numGaussians: numGaussians)
        let loadStart = Date()
        let pipeline = try IdolPipeline(modelsDirectory: root)
        print(String(format: "loaded in %.1f s", Date().timeIntervalSince(loadStart)))
        let input = try RGBAImage(contentsOf: URL(fileURLWithPath: image))
        var noise = SeededNoise(seed: seed)
        var lastStage = ""
        let output = try pipeline.generate(image: input, request: request, noise: &noise,
                                           guidance: guidance == "batched" ? .batched : .sequential,
                                           progress: { p in
            if p.stage.rawValue != lastStage { print("  \(p.stage.rawValue)…"); lastStage = p.stage.rawValue }
        })
        let name = URL(fileURLWithPath: image).deletingPathExtension().lastPathComponent
        try output.write(to: URL(fileURLWithPath: out), name: name)
        let t = output.timings
        print(String(format: "GENERATE OK %@: %d Gaussians — preprocess %.1f s, encode %.1f s, denoise %.1f s, decode %.1f s → %@/%@.ply",
                     name, output.cloud.count, t["preprocess"] ?? 0, t["encode"] ?? 0, t["denoise"] ?? 0, t["decode"] ?? 0, out, name))
    }
}
