// Embedders.swift - sinusoidal scalar and point embeddings (`idol/models/layers.py`)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// π rounded to nearest in fp32, as torch and numpy use it. Swift's `Float.pi` is rounded toward
/// zero (one ulp lower): at angles of ~3000 rad that one ulp moves sin/cos by 2.4e-4 (pitfall 20).
public let piFloat32 = Float(Double.pi)

/// `SinusoidalEmbedder`: `[cos(t·f), sin(t·f)]` with f = exp(−ln(maxPeriod)·k/128), k < 128,
/// optionally ×2π, then Linear(256 → C), SiLU, Linear(C → C) (keys `mlp.0`, `mlp.2`).
/// The denoiser feeds 1000·t; the octree decoder feeds the resolution 1 << level (pitfall 6).
public final class SinusoidalEmbedder: Module {
    @ModuleInfo(key: "mlp") var mlp: [UnaryLayer]
    let frequencyDim: Int
    let maxPeriod: Double
    let twoPi: Bool

    public init(hidden: Int, frequencyDim: Int = 256, maxPeriod: Double = 10_000, twoPi: Bool = false) {
        self.frequencyDim = frequencyDim
        self.maxPeriod = maxPeriod
        self.twoPi = twoPi
        _mlp.wrappedValue = [Linear(frequencyDim, hidden), SiLU(), Linear(hidden, hidden)]
        super.init()
    }

    /// t: [B] → [B, hidden] (fp32 like the reference).
    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        let half = frequencyDim / 2
        // The reference's own fp32 frequencies when tabulated: torch's exp is ~1 ulp off a correctly
        // rounded exp, so this removes one source of divergence at angles up to 256·2π. Its effect on
        // the real octree was below the measurement (T-1.5: the gap there was a fixture bug).
        let freqs: MLXArray
        if let table = SinusoidalTables.frequencies(maxPeriod: maxPeriod, half: half) {
            freqs = MLXArray(table)
        } else {
            freqs = exp(-Float(log(maxPeriod)) * MLXArray(0..<half).asType(.float32) / Float(half))
        }
        var args = t.asType(.float32).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        if twoPi { args = (args * 2) * piFloat32 }  // `args * 2 * torch.pi`: two roundings, in that order
        var h = concatenated([cos(args), sin(args)], axis: -1)
        for layer in mlp { h = layer(h) }
        return h
    }
}

/// `PcdAbsolutePositionEmbedderv2` (both decoders): per axis `[sin(x·f·π), cos(x·f·π)]` with
/// f = 2^linspace(0, 10, C/3/2), zero-padded to C (pitfall 7). No parameters.
public struct PcdPositionEmbedderV2: Sendable {
    public let channels: Int
    let freqs: [Float]

    public init(channels: Int, inChannels: Int = 3, maxRes: Double = 10) {
        self.channels = channels
        let n = channels / inChannels / 2
        // torch.linspace (CPU, float32) fills the first half as start + step·i and the second as
        // end − step·(n − 1 − i), each with one rounding (fused multiply-add). Reproducing that keeps
        // the frequencies bit-identical (checked for n = 8…341); the angles reach ~3000 rad.
        let end = Float(maxRes)
        let step = n > 1 ? end / Float(n - 1) : 0
        let logs = (0..<n).map { i in
            i < n / 2 ? Float(0).addingProduct(step, Float(i)) : end.addingProduct(-step, Float(n - 1 - i))
        }
        freqs = logs.map { pow(Float(2), $0) }
    }

    /// x: [N, 3] → [N, channels].
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let n = x.dim(0)
        let ang = x.asType(.float32).expandedDimensions(axis: -1) * MLXArray(freqs) * piFloat32  // [N, 3, F]
        var embed = concatenated([sin(ang), cos(ang)], axis: -1).reshaped(n, -1)
        let pad = channels - embed.dim(1)
        if pad > 0 { embed = concatenated([embed, MLXArray.zeros([n, pad], dtype: embed.dtype)], axis: 1) }
        return embed
    }
}
