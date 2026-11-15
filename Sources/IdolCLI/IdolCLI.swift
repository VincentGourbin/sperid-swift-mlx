// IdolCLI.swift - `idol`: command line for the Idol image-to-3D port
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import IdolCore
import SperidCore

@main
struct IdolCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "idol",
        abstract: "speridlabs/idol on Apple Silicon (MLX): one image → 3D Gaussian splat.",
        version: IdolCore.version,
        subcommands: [InfoCommand.self, DownloadCommand.self, DecodeCommand.self, DenoiseCommand.self, ParityCommand.self, ProfileCommand.self, ScheduleCommand.self]
    )
}

/// Pinned files, local presence, and header contracts (reads headers only, never the weights).
struct InfoCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "info", abstract: "Show the pinned weights and check the local copies against their contracts.")

    @Option(help: "Models directory (default: $SPERID_MODELS_DIR, else ~/Library/Caches/models).")
    var modelsDir: String?

    func run() throws {
        let root = modelsDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
            ?? ModelsDirectory.resolve()
        print("models directory: \(root.path)")
        let checks: [(HubFile, WeightContract?)] = [
            (IdolWeights.config, nil),
            (IdolWeights.denoiser, IdolWeights.denoiserContract),
            (IdolWeights.octreeDecoder, IdolWeights.octreeDecoderContract),
            (IdolWeights.gaussianDecoder, IdolWeights.gaussianDecoderContract),
            (IdolWeights.birefnet, IdolWeights.birefnetContract),
            (IdolWeights.turboLoRA, IdolWeights.turboLoRAContract),
            (IdolWeights.dinov3Comfy, IdolWeights.dinov3Contract),
            (IdolWeights.flux2VAEComfy, nil),
        ]
        for (file, contract) in checks {
            let url = file.localURL(in: root)
            let size = String(format: "%7.1f MB", Double(file.size) / 1e6)
            guard FileManager.default.fileExists(atPath: url.path) else {
                print("  absent   \(size)  \(file.repoID)/\(file.path)\(file.gated ? "  (gated)" : "")")
                continue
            }
            var status = "present "
            if let contract {
                do {
                    try contract.verify(SafetensorsHeader(contentsOf: url))
                    status = "ok      "
                } catch {
                    status = "MISMATCH"
                    print("  \(error.localizedDescription)")
                }
            }
            print("  \(status) \(size)  \(file.repoID)/\(file.path)")
        }
        let configURL = IdolWeights.config.localURL(in: root)
        if let config = try? IdolConfig.load(from: configURL) {
            print("config: image \(config.imageSize)², octree level \(config.maxVoxelLevel), "
                + "joint tokens \(config.jointTokenCount) (latent \(config.denoiser.qTokenLength) + image "
                + "\(config.conditionTokenCount) + camera 1)")
        }
    }
}

/// The Euler grid of a request, for checking a parity dump by eye.
struct ScheduleCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "schedule", abstract: "Print the time grid (t, network time, turbo phase) of a request.")

    @Option(help: "Sampling steps (SDK default 50, turbo 5).") var steps = 50
    @Option(name: .customLong("rescale-t"), help: "Time shift (SDK default 3.0).") var rescaleT = 3.0

    func run() throws {
        let schedule = FlowSchedule(steps: steps, rescaleT: rescaleT)
        for i in 0..<schedule.steps {
            let (t, next) = schedule.interval(i)
            print(String(format: "%3d  t=%.6f → %.6f  net=%8.3f  turbo=%@", i, t, next,
                         schedule.networkTime(i), schedule.turboPhase(i).rawValue))
        }
    }
}
