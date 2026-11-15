// Examples.swift - the example gallery of the speridlabs/idol Space (images + their settings)
// Copyright 2026 Vincent Gourbin

import Foundation
import IdolCore

struct ExamplePreset: Codable, Identifiable, Hashable {
    let image: String
    let title: String
    let mode: String
    let seed: UInt64
    let steps: Int
    let cfg: Double
    let rescaleT: Double
    let numGaussians: Int
    var id: String { image }

    enum CodingKeys: String, CodingKey {
        case image, title, mode, seed, steps, cfg, numGaussians = "num_gaussians", rescaleT = "rescale_t"
    }

    var request: IdolRequest {
        IdolRequest(mode: IdolCropMode(rawValue: mode) ?? .synthetic, seed: seed, steps: steps, cfgStrength: cfg,
                    rescaleT: rescaleT, numGaussians: numGaussians)
    }
}

/// Downloads `examples/presets.json` and the images of the public Space once, into the models directory.
enum ExampleStore {
    static let base = URL(string: "https://huggingface.co/spaces/speridlabs/idol/resolve/main/")!

    static func folder(_ root: URL) -> URL { root.appendingPathComponent("speridlabs-idol-space") }

    static func load(root: URL) async throws -> [ExamplePreset] {
        let dir = folder(root)
        let presetsURL = dir.appendingPathComponent("examples/presets.json")
        try FileManager.default.createDirectory(at: presetsURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: presetsURL.path) {
            let (data, _) = try await URLSession.shared.data(from: base.appendingPathComponent("examples/presets.json"))
            try data.write(to: presetsURL)
        }
        return try JSONDecoder().decode([ExamplePreset].self, from: Data(contentsOf: presetsURL))
    }

    static func image(_ preset: ExamplePreset, root: URL) async throws -> URL {
        let url = folder(root).appendingPathComponent(preset.image)
        if !FileManager.default.fileExists(atPath: url.path) {
            let (data, _) = try await URLSession.shared.data(from: base.appendingPathComponent(preset.image))
            try data.write(to: url)
        }
        return url
    }
}
