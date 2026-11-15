// PLYWriter.swift - the standard 3D Gaussian Splatting PLY (`Gaussian.save_ply`)
// Copyright 2026 Vincent Gourbin

import Foundation

/// Binary little-endian PLY with the 17 float properties every 3DGS viewer reads. Stored values
/// are the *inverse* activations (logit opacity, log scale), with the reference's clamps so a
/// saturated opacity or a zero scale is never written as ±inf; the z-up object is rotated
/// +90° about x: (x, y, z) → (x, −z, y), q → q_x(90°) ⊗ q. The SDK calls this "y-up", but the object's
/// up (+z) lands on **−y**: the file follows the COLMAP / original 3DGS y-down convention (pitfall 21);
/// a y-up renderer (RealityKit) must turn it half a turn about x.
public enum PLYWriter {
    public static let propertyNames = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]

    public static func header(vertexCount: Int) -> String {
        var lines = ["ply", "format binary_little_endian 1.0", "element vertex \(vertexCount)"]
        lines += propertyNames.map { "property float \($0)" }
        lines.append("end_header")
        return lines.joined(separator: "\n") + "\n"
    }

    /// The 17 floats of one vertex, in file order.
    public static func vertex(_ cloud: GaussianCloud, _ i: Int) -> [Float] {
        let eps: Float = 1e-6
        let x = cloud.xyz[3 * i], y = cloud.xyz[3 * i + 1], z = cloud.xyz[3 * i + 2]
        let o = min(max(cloud.opacity[i], eps), 1 - eps)
        let s = (0..<3).map { log(max(cloud.scaling[3 * i + $0], eps)) }
        let q = rotateYUp((0..<4).map { cloud.rotation[4 * i + $0] })
        return [x, -z, y, 0, 0, 0,
                cloud.featuresDC[3 * i], cloud.featuresDC[3 * i + 1], cloud.featuresDC[3 * i + 2],
                log(o / (1 - o)), s[0], s[1], s[2], q[0], q[1], q[2], q[3]]
    }

    /// Hamilton product (√½, √½, 0, 0) ⊗ q, q = (w, x, y, z).
    static func rotateYUp(_ q: [Float]) -> [Float] {
        let a = Float(0.5).squareRoot()
        let (bw, bx, by, bz) = (q[0], q[1], q[2], q[3])
        return [a * bw - a * bx, a * bx + a * bw, a * by - a * bz, a * bz + a * by]
    }

    public static func data(_ cloud: GaussianCloud) -> Data {
        var data = Data(header(vertexCount: cloud.count).utf8)
        var floats = [Float]()
        floats.reserveCapacity(cloud.count * propertyNames.count)
        for i in 0..<cloud.count { floats += vertex(cloud, i) }
        floats.withUnsafeBufferPointer { buffer in
            // Apple Silicon is little-endian: the buffer is already in file byte order.
            data.append(UnsafeBufferPointer(start: UnsafeRawPointer(buffer.baseAddress!)
                .assumingMemoryBound(to: UInt8.self), count: buffer.count * 4))
        }
        return data
    }

    public static func write(_ cloud: GaussianCloud, to url: URL) throws {
        try data(cloud).write(to: url, options: .atomic)
    }
}
