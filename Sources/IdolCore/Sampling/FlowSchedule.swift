// FlowSchedule.swift - the Euler time grid and guidance of `sampler.py`
// Copyright 2026 Vincent Gourbin

import Foundation

/// Rectified flow from noise (t = 1) to data (t = 0). `sampler.py`:
/// `t = linspace(1, 0, steps + 1)`, then `t ← r·t / (1 + (r − 1)·t)`; one Euler step is
/// `x ← x − v·(t_i − t_{i+1})`, and the network sees `1000·t` (fp32).
public struct FlowSchedule: Sendable, Equatable {
    public let timesteps: [Double]

    public init(steps: Int, rescaleT: Double) {
        precondition(steps >= 1, "steps must be ≥ 1")
        timesteps = (0...steps).map { i in
            let t = 1.0 - Double(i) / Double(steps)
            return rescaleT * t / (1 + (rescaleT - 1) * t)
        }
    }

    public var steps: Int { timesteps.count - 1 }

    /// (t, t_prev) of step `i`, as the sampler loop uses them.
    public func interval(_ i: Int) -> (t: Double, next: Double) { (timesteps[i], timesteps[i + 1]) }

    /// The scalar the denoiser's time embedder receives for step `i`.
    public func networkTime(_ i: Int) -> Float { Float(1000 * timesteps[i]) }

    /// Turbo phase of step `i` (`turbo.py`): the slow expert while t ≥ 0.5, the fast one below.
    public func turboPhase(_ i: Int, threshold: Double = 0.5) -> TurboPhase {
        timesteps[i] >= threshold ? .slow : .fast
    }
}

/// Which distilled LoRA expert is active for a step (`idol-turbo`).
public enum TurboPhase: String, Sendable, Equatable {
    case slow, fast
}
