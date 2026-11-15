// GaussianCloud.swift - activated 3D Gaussians, as renderers and PLY files use them
// Copyright 2026 Vincent Gourbin

import Foundation

/// `gaussian.py` `Gaussian`, activated values in flat row-major Float buffers. The object is
/// z-up inside [-0.5, 0.5]³ (`PLYWriter` turns it y-up). Per Gaussian: xyz 3, colour 3 (degree-0
/// spherical harmonics), scaling 3, rotation 4 (unit quaternion w, x, y, z), opacity 1.
public struct GaussianCloud: Sendable, Equatable {
    public var xyz: [Float]
    public var featuresDC: [Float]
    public var scaling: [Float]
    public var rotation: [Float]
    public var opacity: [Float]

    public var count: Int { opacity.count }

    public init(xyz: [Float], featuresDC: [Float], scaling: [Float], rotation: [Float], opacity: [Float]) {
        precondition(xyz.count == 3 * opacity.count && featuresDC.count == 3 * opacity.count
            && scaling.count == 3 * opacity.count && rotation.count == 4 * opacity.count, "inconsistent buffers")
        self.xyz = xyz
        self.featuresDC = featuresDC
        self.scaling = scaling
        self.rotation = rotation
        self.opacity = opacity
    }

    /// The activations of `Gaussian.__init__`, from the decoder's raw outputs (CPU reference;
    /// the MLX path of `T-1.6` is checked against it):
    /// xyz = raw·1 − 0.5 (aabb [-0.5, 1]), scaling = √(softplus(raw + softplus⁻¹(bias))² + k²),
    /// rotation = normalize(raw + (1, 0, 0, 0)), opacity = σ(raw + σ⁻¹(bias)).
    public static func activate(rawXYZ: [Float], featuresDC: [Float], rawScaling: [Float], rawRotation: [Float],
                                rawOpacity: [Float], scalingBias: Double, opacityBias: Double,
                                minimumKernelSize: Double) -> GaussianCloud {
        let scalingOffset = Float(log(expm1(scalingBias)))          // inverse_softplus
        let opacityOffset = Float(log(opacityBias / (1 - opacityBias)))  // inverse_sigmoid
        let k2 = Float(minimumKernelSize * minimumKernelSize)
        let xyz = rawXYZ.map { $0 - 0.5 }
        let scaling = rawScaling.map { raw -> Float in
            let x = raw + scalingOffset
            let s = x > 20 ? x : log1p(exp(x))  // torch softplus threshold 20
            return (s * s + k2).squareRoot()
        }
        var rotation = rawRotation
        for g in 0..<(rawRotation.count / 4) {
            var q = (0..<4).map { rawRotation[4 * g + $0] }
            q[0] += 1
            let norm = max(q.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1e-12)  // F.normalize eps
            for c in 0..<4 { rotation[4 * g + c] = q[c] / norm }
        }
        let opacity = rawOpacity.map { 1 / (1 + exp(-($0 + opacityOffset))) }
        return GaussianCloud(xyz: xyz, featuresDC: featuresDC, scaling: scaling, rotation: rotation, opacity: opacity)
    }
}
