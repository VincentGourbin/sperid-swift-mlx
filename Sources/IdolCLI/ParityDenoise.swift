// ParityDenoise.swift - `idol parity denoise`: the 50-step CFG trajectory on a real object
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import IdolCore
import MLX
import SperidCore

extension ParityCommand {
    /// Python conditioning + recorded noise → Swift sampler (fp16, the reference precision) → compare the
    /// trajectory at the recorded checkpoints and the final sample, then decode it in Swift (`.ply`).
    func runDenoise(root: URL) throws {
        let f = try loadArrays(url: root.appendingPathComponent("parity/denoiser_avocado.safetensors"))
        let config = try IdolConfig.load(from: IdolWeights.config.localURL(in: root))
        let model = try IdolDenoiser.load(modelsDirectory: root, config: config, blockDType: .float16)
        let request = IdolRequest()
        let schedule = FlowSchedule(steps: request.steps, rescaleT: request.rescaleT)
        let condition = IdolCondition(feature1: f["feature1"]!, feature2: f["feature2"]!)
        var checkpoints: [Int: (MLXArray, MLXArray)] = [:]
        var current = (f["latent_noise"]!, f["camera_noise"]!)
        checkpoints[0] = current
        let start = Date()
        // Run step by step to observe the sample entering steps 10, 25, 49 (one-step schedules would change t).
        let sample = try FlowSampler.sample(
            denoiser: model, latentNoise: current.0, cameraNoise: current.1, condition: condition, schedule: schedule,
            cfgStrength: request.cfgStrength, guidance: guidance == "batched" ? .batched : .sequential,
            observe: { i, x, c in if [9, 24, 48].contains(i) { checkpoints[i + 1] = (x, c) } })
        current = sample
        let seconds = Date().timeIntervalSince(start)
        var worst: Float = 0
        for k in [10, 25, 49] {
            guard let ours = checkpoints[k], let ref = f["traj_\(k)_latent"] else { continue }
            let r = relL2(ours.0, ref)
            worst = max(worst, r)
            print("PARITY INFO denoise.trajectory step \(k) relL2=\(r)")
        }
        let finalLatent = relL2(sample.latent, f["latent"]!)
        let finalCamera = relL2(sample.camera, f["camera"]!)
        print(String(format: "PARITY INFO denoise.sampling %d steps CFG %.1f %@ in %.1f s (load excluded, not a benchmark)",
                     schedule.steps, request.cfgStrength, guidance, seconds))
        let ok = finalLatent < 2e-2
        print("PARITY \(ok ? "OK" : "FAILED") denoise.final latent relL2=\(finalLatent) camera relL2=\(finalCamera)")

        let pose = CameraPose(latent: sample.camera.reshaped(-1).asArray(Float.self))
        let ref = CameraPose(latent: f["camera"]!.reshaped(-1).asArray(Float.self))
        print(String(format: "PARITY INFO denoise.camera swift yaw %.3f pitch %.3f fov %.2f | python yaw %.3f pitch %.3f fov %.2f",
                     pose.yaw, pose.pitch, pose.fov, ref.yaw, ref.pitch, ref.fov))
        let decoder = try IdolDecoder(modelsDirectory: root, blockDType: .float16)
        var noise = SeededNoise(seed: 0)
        let cloud = decoder.decode(latent: sample.latent, numGaussians: request.numGaussians, noise: &noise).cloud
        try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        try PLYWriter.write(cloud, to: URL(fileURLWithPath: "\(outDir)/avocado_swift.ply"))
        print("PARITY INFO denoise.ply \(outDir)/avocado_swift.ply (\(cloud.count) Gaussians)")
        print(ok ? "PARITY OK denoise (final latent relL2 \(finalLatent))" : "PARITY FAILED denoise")
        if !ok { throw ExitCode.failure }
    }
}
