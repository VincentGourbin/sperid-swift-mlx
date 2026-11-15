// CameraPose.swift - the input camera predicted alongside the object (`camera.py`)
// Copyright 2026 Vincent Gourbin

import Foundation

/// Decoded from the 5-channel camera token: direction (3), inverse distance, half width.
public struct CameraPose: Sendable, Equatable, Codable {
    /// Radians.
    public let yaw: Double
    /// Radians.
    public let pitch: Double
    public let radius: Double
    /// Vertical field of view, degrees.
    public let fov: Double

    public init(yaw: Double, pitch: Double, radius: Double, fov: Double) {
        self.yaw = yaw
        self.pitch = pitch
        self.radius = radius
        self.fov = fov
    }

    /// `decode_camera`: an inverse distance under 1e-3 is orthographic (radius 1000).
    public init(latent: [Float]) {
        precondition(latent.count == 5, "camera latent has 5 channels")
        let d = latent[0..<3].map(Double.init)
        let inverseDistance = Double(latent[3]), halfWidth = Double(latent[4])
        let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).squareRoot()
        let (dx, dy, dz) = norm > 1e-6 ? (d[0] / norm, d[1] / norm, d[2] / norm) : (0, 0, 1)
        let r: Double, f: Double
        if inverseDistance < 1e-3 {
            r = 1000
            f = 2 * atan(halfWidth / r) * 180 / .pi
        } else {
            r = 1 / inverseDistance
            f = 2 * atan(halfWidth * inverseDistance) * 180 / .pi
        }
        self.init(yaw: atan2(dx, dy), pitch: asin(min(max(dz, -1), 1)), radius: r, fov: f)
    }
}
