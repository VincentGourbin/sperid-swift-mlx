// IdolRequest.swift - one generation request, with the SDK's defaults and bounds
// Copyright 2026 Vincent Gourbin

import Foundation
import SperidCore

/// How much room the square crop leaves around the object (`preprocess.py`, `CROP_MARGIN`).
public enum IdolCropMode: String, Sendable, CaseIterable, Codable {
    /// Renders and generated images of one object: the object spans 1/1.2 of the frame.
    case synthetic
    /// Photographs: the object spans 1/1.7 of the frame.
    case real

    public var margin: Double {
        switch self {
        case .synthetic: 1.2
        case .real: 1.7
        }
    }
}

/// The parameters of `IdolPipeline.__call__` (`pipeline.py`), same names, same defaults.
public struct IdolRequest: Sendable, Equatable, Codable {
    public var mode: IdolCropMode
    public var seed: UInt64
    public var steps: Int
    /// Classifier-free guidance w: v = (1 + w)·v(cond) − w·v(uncond). 0 disables the second pass.
    public var cfgStrength: Double
    /// Time shift of the Euler grid (more steps at high noise when > 1).
    public var rescaleT: Double
    public var numGaussians: Int
    /// Overrides the mode's margin when set.
    public var cropMargin: Double?

    public static let minimumGaussians = 32_768
    public static let maximumGaussians = 1_048_576

    public init(mode: IdolCropMode = .synthetic, seed: UInt64 = 42, steps: Int = 50, cfgStrength: Double = 3.0,
                rescaleT: Double = 3.0, numGaussians: Int = 262_144, cropMargin: Double? = nil) {
        self.mode = mode
        self.seed = seed
        self.steps = steps
        self.cfgStrength = cfgStrength
        self.rescaleT = rescaleT
        self.numGaussians = numGaussians
        self.cropMargin = cropMargin
    }

    /// The turbo defaults (`speridlabs/idol-turbo` README): 5 guidance-free steps.
    public static var turbo: IdolRequest { IdolRequest(steps: 5, cfgStrength: 0) }

    public var effectiveCropMargin: Double { cropMargin ?? mode.margin }

    /// Anchor points the octree decoder places: num_gaussians / 32, at least 1 (`pipeline.decode`).
    public func anchorCount(gaussiansPerAnchor: Int = 32) -> Int { max(1, numGaussians / gaussiansPerAnchor) }

    /// Throws `.invalidRequest` for values the SDK documents as out of range.
    public func validate() throws {
        guard steps >= 1 else { throw SperidError.invalidRequest("steps must be ≥ 1 (got \(steps))") }
        guard cfgStrength >= 0 else { throw SperidError.invalidRequest("cfgStrength must be ≥ 0") }
        guard rescaleT > 0 else { throw SperidError.invalidRequest("rescaleT must be > 0") }
        // The README documents 32,768…1,048,576; the SDK itself only rounds down to a multiple
        // of 32 (`anchorCount`), so a non-multiple is accepted, not rejected.
        guard (Self.minimumGaussians...Self.maximumGaussians).contains(numGaussians) else {
            throw SperidError.invalidRequest(
                "numGaussians must be in \(Self.minimumGaussians)…\(Self.maximumGaussians) (got \(numGaussians))")
        }
        if let m = cropMargin, m < 1 { throw SperidError.invalidRequest("cropMargin must be ≥ 1") }
    }
}
