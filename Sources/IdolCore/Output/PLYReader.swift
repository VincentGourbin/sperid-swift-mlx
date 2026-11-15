// PLYReader.swift - read a standard 3DGS PLY back into a GaussianCloud (inverse of PLYWriter)
// Copyright 2026 Vincent Gourbin

import Foundation
import SperidCore

/// Reads binary little-endian 3DGS PLY files whose vertex properties include x, y, z, f_dc_0…2,
/// opacity, scale_0…2, rot_0…3 as float (Idol's and the SDK's files; extra properties such as normals
/// or higher SH bands are skipped). Values are turned back into the activated, z-up `GaussianCloud`.
public enum PLYReader {
    public static func read(_ url: URL) throws -> GaussianCloud {
        let data = try Data(contentsOf: url)
        guard let end = data.range(of: Data("end_header\n".utf8)) else {
            throw SperidError.invalidRequest("\(url.lastPathComponent): not a PLY file")
        }
        let header = String(decoding: data[..<end.lowerBound], as: UTF8.self)
        guard header.contains("format binary_little_endian") else {
            throw SperidError.invalidRequest("\(url.lastPathComponent): only binary little-endian PLY is supported")
        }
        var count = 0
        var names: [String] = []
        for line in header.split(separator: "\n") {
            let parts = line.split(separator: " ")
            if parts.count == 3, parts[0] == "element", parts[1] == "vertex" { count = Int(parts[2]) ?? 0 }
            if parts.count == 3, parts[0] == "property" {
                guard parts[1] == "float" else { throw SperidError.invalidRequest("non-float property \(parts[2])") }
                names.append(String(parts[2]))
            }
        }
        let stride = names.count
        let index = Dictionary(uniqueKeysWithValues: names.enumerated().map { ($1, $0) })
        let required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                        "rot_0", "rot_1", "rot_2", "rot_3"]
        guard required.allSatisfy({ index[$0] != nil }) else {
            throw SperidError.invalidRequest("\(url.lastPathComponent): missing 3DGS properties")
        }
        let body = data[end.upperBound...]
        guard body.count >= count * stride * 4 else { throw SperidError.invalidRequest("truncated PLY") }
        var floats = [Float](repeating: 0, count: count * stride)
        _ = floats.withUnsafeMutableBytes { body.copyBytes(to: $0, count: count * stride * 4) }
        var cloud = GaussianCloud(xyz: [], featuresDC: [], scaling: [], rotation: [], opacity: [])
        cloud.xyz.reserveCapacity(3 * count)
        let a = Float(0.5).squareRoot()
        for i in 0..<count {
            func v(_ n: String) -> Float { floats[i * stride + index[n]!] }
            // y-up → z-up: (x, y, z)_file = (x, −z, y)_object  ⇒  object = (x, z_file, −y_file)
            cloud.xyz += [v("x"), v("z"), -v("y")]
            cloud.featuresDC += [v("f_dc_0"), v("f_dc_1"), v("f_dc_2")]
            cloud.opacity.append(1 / (1 + exp(-v("opacity"))))
            cloud.scaling += [exp(v("scale_0")), exp(v("scale_1")), exp(v("scale_2"))]
            // q_file = (√½, √½, 0, 0) ⊗ q  ⇒  q = (√½, −√½, 0, 0) ⊗ q_file
            let (bw, bx, by, bz) = (v("rot_0"), v("rot_1"), v("rot_2"), v("rot_3"))
            var q = [a * bw + a * bx, a * bx - a * bw, a * by + a * bz, a * bz - a * by]
            let n = max((q.reduce(0) { $0 + $1 * $1 }).squareRoot(), 1e-12)
            q = q.map { $0 / n }
            cloud.rotation += q
        }
        return cloud
    }
}
