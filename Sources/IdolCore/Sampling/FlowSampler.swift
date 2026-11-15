// FlowSampler.swift - Euler integration with classifier-free guidance (`idol/sampler.py`)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import SperidCore

/// The image condition of one object: DINOv3 tokens [1, Lc, 1280] and FLUX.2 VAE tokens [1, Lc, 128].
public struct IdolCondition {
    public let feature1: MLXArray
    public let feature2: MLXArray

    public init(feature1: MLXArray, feature2: MLXArray) {
        self.feature1 = feature1
        self.feature2 = feature2
    }

    /// The negative condition: zeros **before** the projections, so both biases still apply (pitfall 9).
    public var negative: IdolCondition {
        IdolCondition(feature1: MLXArray.zeros(like: feature1), feature2: MLXArray.zeros(like: feature2))
    }
}

public enum FlowSampler {
    public enum Guidance: Sendable {
        /// cond and neg as one batch-2 forward (more memory, one dispatch).
        case batched
        /// two batch-1 forwards per step.
        case sequential
    }

    /// From noise (t = 1) to data (t = 0): `x ← x − v·(t − t_next)` on both the latent and the camera,
    /// v = (1 + w)·v(cond) − w·v(neg) when w > 0. One synchronisation per step (IT-2); cancellation is
    /// checked before every step and throws `CancellationError`.
    public static func sample(denoiser: IdolDenoiser, latentNoise: MLXArray, cameraNoise: MLXArray,
                              condition: IdolCondition, schedule: FlowSchedule, cfgStrength: Double,
                              guidance: Guidance = .sequential, isCancelled: () -> Bool = { false },
                              observe: (_ step: Int, _ latent: MLXArray, _ camera: MLXArray) -> Void = { _, _, _ in })
        throws -> (latent: MLXArray, camera: MLXArray) {
        var x = latentNoise.asType(.float32)
        var cam = cameraNoise.asType(.float32)
        let neg = condition.negative
        let w = Float(cfgStrength)
        for i in 0..<schedule.steps {
            if isCancelled() { throw CancellationError() }
            let (t, next) = schedule.interval(i)
            let netT = MLXArray([schedule.networkTime(i)])
            var vx: MLXArray, vc: MLXArray
            if w > 0, guidance == .batched {
                let v = denoiser(latent: concatenated([x, x], axis: 0), camera: concatenated([cam, cam], axis: 0),
                                 t: concatenated([netT, netT], axis: 0),
                                 feature1: concatenated([condition.feature1, neg.feature1], axis: 0),
                                 feature2: concatenated([condition.feature2, neg.feature2], axis: 0))
                vx = (1 + w) * v.latent[0..<1] - w * v.latent[1..<2]
                vc = (1 + w) * v.camera[0..<1] - w * v.camera[1..<2]
            } else {
                let v = denoiser(latent: x, camera: cam, t: netT, feature1: condition.feature1, feature2: condition.feature2)
                (vx, vc) = (v.latent, v.camera)
                if w > 0 {
                    let u = denoiser(latent: x, camera: cam, t: netT, feature1: neg.feature1, feature2: neg.feature2)
                    vx = (1 + w) * vx - w * u.latent
                    vc = (1 + w) * vc - w * u.camera
                }
            }
            let dt = Float(t - next)
            x = x - vx * dt
            cam = cam - vc * dt
            eval(x, cam)
            observe(i, x, cam)
        }
        return (x, cam)
    }
}
