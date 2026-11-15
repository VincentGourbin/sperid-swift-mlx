// GaussianAssembly.swift - raw decoder outputs → activated Gaussians, in MLX (`to_representation` + `Gaussian`)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

public enum GaussianAssembly {
    /// Raw channels per Gaussian: offset 3, colour 3, scale 3, rotation 4, opacity 1, offset scale 1.
    public static let channelsPerGaussian = 15
    /// The reference layout: one contiguous block per attribute, each [perAnchor × width].
    static let layout: [(name: String, width: Int)] = [
        ("_xyz", 3), ("_features_dc", 3), ("_scaling", 3), ("_rotation", 4), ("_opacity", 1), ("_offset_scale", 1),
    ]

    /// torch's `inverse_softplus` / `inverse_sigmoid` of the biases, in fp32 like the reference.
    static func scalingOffset(_ bias: Double) -> Float {
        let x = Float(bias)
        return x + log(-expm1(-x))
    }
    static func opacityOffset(_ bias: Double) -> Float {
        let x = Float(bias)
        return log(x / (1 - x))
    }

    /// torch `F.softplus` (beta 1, threshold 20).
    static func softplus(_ x: MLXArray) -> MLXArray { which(x .> 20, x, log1p(exp(x))) }

    /// - Parameters: anchors [1, L, 3]; features [1, L, 480] (one object).
    /// - Returns: the activated cloud of L × perAnchor Gaussians, anchor-major.
    public static func assemble(anchors: MLXArray, features: MLXArray, decoder: GaussianDecoder) -> GaussianCloud {
        let rep = decoder.representation
        let perAnchor = rep.numGaussians
        let l = features.dim(1)
        let f = features.asType(.float32)[0]
        var slices: [String: MLXArray] = [:]
        var start = 0
        for (name, width) in layout {
            let size = perAnchor * width
            slices[name] = f[0..., start..<(start + size)].reshaped(l, perAnchor, width)
            start += size
        }
        func lr(_ name: String) -> Float { Float(rep.lr[name] ?? 1) }
        let offsetScale = softplus(slices["_offset_scale"]! + decoder.baseOffsetScale.asType(.float32))
        var offset = slices["_xyz"]! * lr("_xyz") + decoder.pointsOffsetPerturbation.asType(.float32)
        offset = tanh(offset) * 0.5 * Float(rep.perturbeSize) * offsetScale
        let rawXYZ = (offset + anchors.asType(.float32)[0].expandedDimensions(axis: 1)).reshaped(-1, 3)
        let n = l * perAnchor
        let colour = (slices["_features_dc"]! * lr("_features_dc")).reshaped(n, 3)
        let rawScale = (slices["_scaling"]! * lr("_scaling")).reshaped(n, 3)
        let rawRotation = (slices["_rotation"]! * lr("_rotation")).reshaped(n, 4)
        let rawOpacity = (slices["_opacity"]! * lr("_opacity")).reshaped(n, 1)

        // `Gaussian.__init__`: aabb [-0.5, 1] → xyz − 0.5; scale with the 3D filter; unit quaternion.
        let xyz = rawXYZ - 0.5
        let s = softplus(rawScale + scalingOffset(rep.scalingBias))
        let k = Float(rep.filterKernelSize3D)
        let scaling = sqrt(s * s + k * k)
        let q = rawRotation + MLXArray([Float(1), 0, 0, 0])
        let norm = maximum(sqrt(sum(q * q, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        let rotation = q / norm
        let opacity = sigmoid(rawOpacity + opacityOffset(rep.opacityBias))
        eval(xyz, colour, scaling, rotation, opacity)
        return GaussianCloud(xyz: xyz.asArray(Float.self), featuresDC: colour.asArray(Float.self),
                             scaling: scaling.asArray(Float.self), rotation: rotation.asArray(Float.self),
                             opacity: opacity.reshaped(-1).asArray(Float.self))
    }
}
