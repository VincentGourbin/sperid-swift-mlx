// DenoiseCommand.swift - `idol denoise`: image tokens → object latent (+ camera, + optional .ply)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import IdolCore
import MLX
import SperidCore

struct DenoiseCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "denoise",
        abstract: "Run the flow sampler on image tokens (feature1, feature2) and write the latent, the camera, optionally the .ply.")

    @Option(help: "A .safetensors file with `feature1` [1, 4101, 1280] and `feature2` [1, 4101, 128].") var cond: String
    @Option(help: "Sampling steps.") var steps = 50
    @Option(help: "Classifier-free guidance strength (0 = one evaluation per step).") var cfg = 3.0
    @Option(name: .customLong("rescale-t"), help: "Time shift.") var rescaleT = 3.0
    @Option(help: "Seed of the starting noise (Swift stream, not torch's).") var seed: UInt64 = 42
    @Option(help: "Replay `latent_noise` / `camera_noise` from this .safetensors instead.") var noise: String?
    @Option(help: "Guidance: sequential | batched.") var guidance = "sequential"
    @Option(help: "Output .safetensors (latent, camera).") var out: String
    @Option(help: "Also decode the latent into this .ply.") var ply: String?
    @Option(help: "Models directory.") var modelsDir: String?

    func run() throws {
        let root = modelsDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? ModelsDirectory.resolve()
        let request = IdolRequest(seed: seed, steps: steps, cfgStrength: cfg, rescaleT: rescaleT)
        try request.validate()
        let c = try loadArrays(url: URL(fileURLWithPath: cond))
        guard let f1 = c["feature1"], let f2 = c["feature2"] else {
            throw SperidError.invalidRequest("\(cond) needs `feature1` and `feature2`")
        }
        let config = try IdolConfig.load(from: IdolWeights.config.localURL(in: root))
        let model = try IdolDenoiser.load(modelsDirectory: root, config: config)
        let d = config.denoiser
        let latentNoise: MLXArray, cameraNoise: MLXArray
        if let noise {
            var replay = ReplayNoise(arrays: try loadArrays(url: URL(fileURLWithPath: noise)))
            latentNoise = replay.latentNoise(shape: [1, d.qTokenLength, d.inChannels])
            cameraNoise = replay.cameraNoise(shape: [1, 1, d.camChannels])
        } else {
            var fresh = SeededNoise(seed: seed)
            latentNoise = fresh.latentNoise(shape: [1, d.qTokenLength, d.inChannels])
            cameraNoise = fresh.cameraNoise(shape: [1, 1, d.camChannels])
        }
        let start = Date()
        let sample = try FlowSampler.sample(
            denoiser: model, latentNoise: latentNoise, cameraNoise: cameraNoise,
            condition: IdolCondition(feature1: f1, feature2: f2),
            schedule: FlowSchedule(steps: steps, rescaleT: rescaleT), cfgStrength: cfg,
            guidance: guidance == "batched" ? .batched : .sequential)
        try MLX.save(arrays: ["latent": sample.latent, "camera": sample.camera], url: URL(fileURLWithPath: out))
        let pose = CameraPose(latent: sample.camera.reshaped(-1).asArray(Float.self))
        print(String(format: "DENOISE OK %d steps, cfg %.1f, %.1f s → %@ (camera yaw %.3f pitch %.3f fov %.1f°)",
                     steps, cfg, Date().timeIntervalSince(start), out, pose.yaw, pose.pitch, pose.fov))
        if let ply {
            let decoder = try IdolDecoder(modelsDirectory: root)
            var fresh = SeededNoise(seed: seed)
            let cloud = decoder.decode(latent: sample.latent, numGaussians: request.numGaussians, noise: &fresh).cloud
            try PLYWriter.write(cloud, to: URL(fileURLWithPath: ply))
            print("PLY \(cloud.count) Gaussians → \(ply)")
        }
    }
}
