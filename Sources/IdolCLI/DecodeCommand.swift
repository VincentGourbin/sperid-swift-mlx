// DecodeCommand.swift - `idol decode`: a latent → `.ply`
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import IdolCore
import MLX
import SperidCore

struct DecodeCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decode", abstract: "Decode an object latent [1, 8192, 16] into a 3DGS .ply.")

    @Option(help: "A .safetensors file holding the latent.") var latent: String
    @Option(help: "Tensor name of the latent in that file.") var key = "latent"
    @Option(help: "Gaussians (32 768 … 1 048 576).") var numGaussians = 262_144
    @Option(help: "Replay the draws recorded in this .safetensors (parity); default: fresh draws from --seed.")
    var noise: String?
    @Option(help: "Prefix of the recorded draws inside --noise (e.g. `fp32.`).") var noisePrefix = ""
    @Option(help: "Seed of the fresh draws.") var seed: UInt64 = 42
    @Flag(help: "fp32 blocks instead of the reference's fp16.") var fp32 = false
    @Option(help: "Output .ply.") var out: String
    @Option(help: "Models directory (default: $SPERID_MODELS_DIR, else ~/Library/Caches/models).")
    var modelsDir: String?

    func run() throws {
        let root = modelsDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
            ?? ModelsDirectory.resolve()
        try IdolRequest(numGaussians: numGaussians).validate()
        let arrays = try loadArrays(url: URL(fileURLWithPath: latent))
        guard let z = arrays[key] else { throw SperidError.invalidRequest("no tensor `\(key)` in \(latent)") }
        let decoder = try IdolDecoder(modelsDirectory: root, blockDType: fp32 ? .float32 : .float16)
        let start = Date()
        let output: IdolDecoder.Output
        if let noise {
            var replay = ReplayNoise(arrays: try loadArrays(url: URL(fileURLWithPath: noise)), prefix: noisePrefix)
            output = decoder.decode(latent: z, numGaussians: numGaussians, noise: &replay)
            if !replay.mismatches.isEmpty { print("note: replay diverged: \(replay.mismatches)") }
        } else {
            var fresh = SeededNoise(seed: seed)
            output = decoder.decode(latent: z, numGaussians: numGaussians, noise: &fresh)
        }
        try PLYWriter.write(output.cloud, to: URL(fileURLWithPath: out))
        print(String(format: "DECODE OK %d Gaussians, cells per level %@, %.1f s → %@", output.cloud.count,
                     output.levels.map { String($0.cells) }.joined(separator: " "), Date().timeIntervalSince(start), out))
    }
}
