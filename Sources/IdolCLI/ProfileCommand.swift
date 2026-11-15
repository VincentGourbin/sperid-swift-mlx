// ProfileCommand.swift - `idol profile denoiser`: one denoiser forward at the real sequence length
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import IdolCore
import MLX
import MLXProfiler
import SperidCore

/// T-2.5: time and peak memory of one forward at L = 8192 + 4101 + 1, batch 1 (sequential CFG) or 2
/// (batched CFG), through swift-mlx-profiler. Inputs are random with the real shapes (the cost does
/// not depend on the values). Follow CLAUDE.md "Performance work" before trusting a number.
struct ProfileCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "profile", abstract: "Profile one stage (denoiser) with swift-mlx-profiler.")

    @Argument(help: "Stage: denoiser.") var stage: String
    @Option(help: "Batch: 1 (sequential CFG) or 2 (batched CFG).") var batch = 1
    @Option(help: "Measured forwards after one warm-up.") var runs = 5
    @Flag(help: "fp32 blocks instead of fp16.") var fp32 = false
    @Option(help: "Output directory for trace.json.") var out = ".local-runs/bench.noindex/profile-denoiser"
    @Option(help: "Models directory.") var modelsDir: String?

    func run() throws {
        guard stage == "denoiser" || stage == "denoiser-ops" else { throw ValidationError("unknown stage \(stage) (denoiser | denoiser-ops)") }
        let root = modelsDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? ModelsDirectory.resolve()
        let config = try IdolConfig.load(from: IdolWeights.config.localURL(in: root))
        let session = ProfilingSession(config: .singleRun)
        session.title = "idol profile denoiser"
        session.metadata = ["batch": "\(batch)", "blocks": fp32 ? "fp32" : "fp16", "tokens": "\(config.jointTokenCount)"]
        MLXProfiler.shared.enable()
        MLXProfiler.shared.activeSession = session

        session.beginPhase("1. Load", category: .modelLoad)
        let model = try IdolDenoiser.load(modelsDirectory: root, config: config, blockDType: fp32 ? .float32 : .float16)
        session.endPhase("1. Load", category: .modelLoad)

        if stage == "denoiser-ops" {
            let ops = DenoiserOps(model: model, runs: runs)
            let rows = ops.run(tokens: config.jointTokenCount)
            session.finish()
            print("OPS one joint block, \(config.jointTokenCount) tokens, fp16 blocks (median of \(runs), warm-up excluded):")
            let whole = rows.last!.1
            for (name, ms) in rows { print(String(format: "  %-40@ %8.2f ms  %5.1f %%  ×24 = %6.0f ms", name as NSString, ms, 100 * ms / whole, 24 * ms)) }
            return
        }
        let lc = config.conditionTokenCount, d = config.denoiser
        let key = MLXRandom.key(0)
        let z = MLXRandom.normal([batch, d.qTokenLength, d.inChannels], key: key)
        let cam = MLXRandom.normal([batch, 1, d.camChannels], key: key)
        let f1 = MLXRandom.normal([batch, lc, d.condChannels], key: key)
        let f2 = MLXRandom.normal([batch, lc, d.cond2Channels], key: key)
        let t = MLXArray([Float](repeating: 1000, count: batch))
        eval(z, cam, f1, f2, t)

        Memory.peakMemory = 0  // setter resets the peak (mlx_reset_peak_memory)
        for i in 0...runs {
            let start = DispatchTime.now().uptimeNanoseconds
            let v = model(latent: z, camera: cam, t: t, feature1: f1, feature2: f2)
            eval(v.latent, v.camera)
            let us = (DispatchTime.now().uptimeNanoseconds - start) / 1000
            if i > 0 { session.recordStep(index: i - 1, total: runs, durationUs: us) }  // forward 0 = warm-up
        }
        session.metadata["peak_mlx_gb"] = String(format: "%.2f", Double(Memory.peakMemory) / 1e9)
        session.finish()
        print(session.generateReport())
        print(String(format: "PROFILE denoiser batch=%d peak_mlx=%.2f GB", batch, Double(Memory.peakMemory) / 1e9))
        try FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
        try ChromeTraceExporter.export(session: session).write(to: URL(fileURLWithPath: out).appendingPathComponent("trace-b\(batch).json"))
    }
}
