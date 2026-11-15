// TinyParity.swift - access to parity/tiny_layers.safetensors and comparison helpers
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

enum TinyParity {
    static let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()

    nonisolated(unsafe) static let arrays: [String: MLXArray] = {
        let url = repoRoot.appendingPathComponent("parity/tiny_layers.safetensors")
        return (try? loadArrays(url: url)) ?? [:]
    }()

    nonisolated(unsafe) static let denoiser: [String: MLXArray] = {
        let url = repoRoot.appendingPathComponent("parity/tiny_denoiser.safetensors")
        return (try? loadArrays(url: url)) ?? [:]
    }()

    static func tensor(_ key: String) -> MLXArray {
        guard let a = arrays[key] ?? denoiser[key] else { fatalError("missing fixture tensor \(key)") }
        return a
    }

    /// max |a − b| / max(max |b|, 1e-6).
    static func relMax(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        let scale = abs(b.asType(.float32)).max().item(Float.self)
        return diff / max(scale, 1e-6)
    }
}
