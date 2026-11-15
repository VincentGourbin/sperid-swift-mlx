// ParityCommand.swift - `idol parity <component>`: Swift against the Python fixtures
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import IdolCore
import MLX
import SperidCore

struct ParityCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "parity", abstract: "Check a component against the Python fixtures (prints PARITY OK|FAILED).")

    @Argument(help: "Component: decode | denoise.") var component: String
    @Option(help: "denoise: guidance mode (sequential | batched).") var guidance = "sequential"
    @Option(help: "Models directory (default: $SPERID_MODELS_DIR, else ~/Library/Caches/models).")
    var modelsDir: String?
    @Option(help: "Where the Swift .ply files go.") var outDir = ".local-runs/parity"

    func run() throws {
        let root = modelsDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
            ?? ModelsDirectory.resolve()
        if component == "denoise" { return try runDenoise(root: root) }
        guard component == "decode" else { throw ValidationError("unknown component \(component) (decode | denoise)") }
        let f = try loadArrays(url: root.appendingPathComponent("parity/decoders.safetensors"))
        try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        var ok = true

        // 1. fp32 end to end, recorded draws: anchors bit for bit, Gaussian features to 1e-4.
        let d32 = try IdolDecoder(modelsDirectory: root, blockDType: .float32)
        var replay = ReplayNoise(arrays: f, prefix: "fp32.")
        let expected = f["fp32.anchors"]!
        let out32 = d32.decode(latent: f["latent"]!, numGaussians: expected.dim(1) * 32, noise: &replay)
        let exact = replay.mismatches.isEmpty && out32.anchors.asArray(Float.self) == expected.asArray(Float.self)
        print("PARITY \(exact ? "OK" : "FAILED") decode.anchors.fp32 exact=\(exact) cells=\(out32.levels.map(\.cells))")
        ok = ok && exact
        let features = d32.gaussian(f["anchors"]!, latent: f["latent"]!)
        let rel = relL2(features, f["fp32.features"]!)
        print("PARITY \(rel < 1e-4 ? "OK" : "FAILED") decode.gaussian.fp32 relL2=\(rel)")
        ok = ok && rel < 1e-4

        // 2. The PLY writer on the reference's own cloud, for compare_ply.py against save_ply.
        let reference = GaussianCloud(
            xyz: f["xyz"]!.reshaped(-1).asArray(Float.self), featuresDC: f["features_dc"]!.reshaped(-1).asArray(Float.self),
            scaling: f["scaling"]!.reshaped(-1).asArray(Float.self), rotation: f["rotation"]!.reshaped(-1).asArray(Float.self),
            opacity: f["opacity"]!.reshaped(-1).asArray(Float.self))
        try PLYWriter.write(reference, to: URL(fileURLWithPath: "\(outDir)/decode_reference_cloud.ply"))

        // 3. The full fp16 decode of the same latent, fresh draws: the file to look at (G-1).
        let d16 = try IdolDecoder(modelsDirectory: root, blockDType: .float16)
        var fresh = SeededNoise(seed: 0)
        let out16 = d16.decode(latent: f["latent"]!, numGaussians: 262_144, noise: &fresh)
        try PLYWriter.write(out16.cloud, to: URL(fileURLWithPath: "\(outDir)/decode_swift_fp16.ply"))
        print("PARITY INFO decode.swift_fp16 \(out16.cloud.count) Gaussians cells=\(out16.levels.map(\.cells))")
        print(ok ? "PARITY OK decode (fp32 anchors exact, Gaussian fp32 relL2 \(rel))" : "PARITY FAILED decode")
        if !ok { throw ExitCode.failure }
    }

    func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        (sqrt(sum(square(a.asType(.float32) - b))) / sqrt(sum(square(b)))).item(Float.self)
    }
}
