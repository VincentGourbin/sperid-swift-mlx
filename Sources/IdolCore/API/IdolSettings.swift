// IdolSettings.swift - what an app asks for: the Space's settings, with two presets
// Copyright 2026 Vincent Gourbin

import Foundation

/// The settings of one generation, named like the speridlabs/idol Space.
public struct IdolSettings: Sendable, Equatable, Codable {
    /// `.synthetic` for renders and generated images (tight crop), `.real` for photographs.
    public var mode: IdolCropMode = .synthetic
    /// `nil` draws a random seed (returned in the result).
    public var seed: UInt64? = 42
    /// idol-turbo: 5 guidance-free steps with the phase LoRAs (≈ 20 s on an M3 Max). `false`: the base model.
    public var turbo = true
    /// Base model only (turbo always runs 5 steps without guidance).
    public var steps = 50
    public var guidance = 3.0
    /// The Space's "Time shift".
    public var timeShift = 3.0
    /// 65 536, 131 072, 262 144, 524 288 or 1 048 576 (any value in 32 768…1 048 576 works).
    public var gaussians = 262_144

    public init(mode: IdolCropMode = .synthetic, seed: UInt64? = 42, turbo: Bool = true, steps: Int = 50,
                guidance: Double = 3.0, timeShift: Double = 3.0, gaussians: Int = 262_144) {
        self.mode = mode
        self.seed = seed
        self.turbo = turbo
        self.steps = steps
        self.guidance = guidance
        self.timeShift = timeShift
        self.gaussians = gaussians
    }

    /// 5 steps, no guidance: the fast preview (idol-turbo).
    public static let turbo = IdolSettings()
    /// The reference: 50 steps, guidance 3 (≈ 8 min on an M3 Max).
    public static let quality = IdolSettings(turbo: false)

    func request(seed: UInt64) -> IdolRequest {
        IdolRequest(mode: mode, seed: seed, steps: turbo ? 5 : steps, cfgStrength: turbo ? 0 : guidance,
                    rescaleT: timeShift, numGaussians: gaussians)
    }
}
