// GLBWriter.swift - binary glTF 2.0 of a SplatMesh (positions, normals, COLOR_0, triangles)
// Copyright 2026 Vincent Gourbin

import Foundation

public enum GLBWriter {
    /// sRGB → linear (glTF vertex colours are linear).
    static func linear(_ c: Float) -> Float { c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }

    public static func data(_ mesh: SplatMesh) -> Data {
        var bin = Data()
        func append<T>(_ values: [T]) { values.withUnsafeBufferPointer { bin.append(UnsafeBufferPointer(start: UnsafeRawPointer($0.baseAddress!).assumingMemoryBound(to: UInt8.self), count: $0.count * MemoryLayout<T>.stride)) } }
        let n = mesh.positions.count
        let positions = mesh.positions.flatMap { [$0.x, $0.y, $0.z] }
        let normals = mesh.normals.flatMap { [$0.x, $0.y, $0.z] }
        let colors = mesh.colors.flatMap { [linear($0.x), linear($0.y), linear($0.z)] }
        let indices = mesh.quads.flatMap { [$0.x, $0.y, $0.z, $0.x, $0.z, $0.w] }
        let offsets = [0, positions.count * 4, (positions.count + normals.count) * 4, (positions.count + normals.count + colors.count) * 4]
        append(positions); append(normals); append(colors); append(indices)
        while bin.count % 4 != 0 { bin.append(0) }
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        for p in mesh.positions { lo = pointwiseMin(lo, p); hi = pointwiseMax(hi, p) }
        let json: [String: Any] = [
            "asset": ["version": "2.0", "generator": "sperid-swift-mlx (Idol)"],
            "scene": 0, "scenes": [["nodes": [0]]], "nodes": [["mesh": 0, "name": "Idol"]],
            "meshes": [["name": "Idol", "primitives": [[
                "attributes": ["POSITION": 0, "NORMAL": 1, "COLOR_0": 2], "indices": 3, "material": 0, "mode": 4,
            ]]]],
            "materials": [["name": "vertex colours", "doubleSided": true,
                           "pbrMetallicRoughness": ["baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1]]],
            "buffers": [["byteLength": bin.count]],
            "bufferViews": [
                ["buffer": 0, "byteOffset": offsets[0], "byteLength": positions.count * 4, "target": 34962],
                ["buffer": 0, "byteOffset": offsets[1], "byteLength": normals.count * 4, "target": 34962],
                ["buffer": 0, "byteOffset": offsets[2], "byteLength": colors.count * 4, "target": 34962],
                ["buffer": 0, "byteOffset": offsets[3], "byteLength": indices.count * 4, "target": 34963],
            ],
            "accessors": [
                ["bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
                 "min": [lo.x, lo.y, lo.z], "max": [hi.x, hi.y, hi.z]],
                ["bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3"],
                ["bufferView": 2, "componentType": 5126, "count": n, "type": "VEC3"],
                ["bufferView": 3, "componentType": 5125, "count": indices.count, "type": "SCALAR"],
            ],
        ]
        var jsonData = try! JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
        while jsonData.count % 4 != 0 { jsonData.append(0x20) }
        var out = Data()
        func u32(_ v: Int) { var x = UInt32(v).littleEndian; out.append(Data(bytes: &x, count: 4)) }
        u32(0x4654_6C67); u32(2); u32(12 + 8 + jsonData.count + 8 + bin.count)    // "glTF", version 2, length
        u32(jsonData.count); u32(0x4E4F_534A); out.append(jsonData)                 // JSON chunk
        u32(bin.count); u32(0x004E_4942); out.append(bin)                           // BIN chunk
        return out
    }

    public static func write(_ mesh: SplatMesh, to url: URL) throws { try data(mesh).write(to: url, options: .atomic) }
}
