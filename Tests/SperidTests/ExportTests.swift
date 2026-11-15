// ExportTests.swift - mesh extraction and the GLB / USDZ containers (weight-free)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import IdolCore

@Suite("Export")
struct ExportTests {
    /// A ball of Gaussians: the extracted surface must be closed-ish, non-empty, inside the bounds.
    static func ball() -> GaussianCloud {
        var xyz: [Float] = [], dc: [Float] = [], sc: [Float] = [], rot: [Float] = [], op: [Float] = []
        for i in 0..<4000 {
            let t = Float(i) * 2.399963, z = 1 - 2 * (Float(i) + 0.5) / 4000, r = (1 - z * z).squareRoot()
            xyz += [0.3 * r * cos(t), 0.3 * r * sin(t), 0.3 * z]
            dc += [1, -1, 0]; sc += [0.02, 0.02, 0.02]; rot += [1, 0, 0, 0]; op.append(0.9)
        }
        return GaussianCloud(xyz: xyz, featuresDC: dc, scaling: sc, rotation: rot, opacity: op)
    }

    @Test func extractsAColouredSurface() {
        let mesh = MeshExtraction.extract(Self.ball(), options: .init(resolution: 64))
        #expect(mesh.quads.count > 1000)
        #expect(mesh.positions.allSatisfy { abs($0.x) < 0.45 && abs($0.y) < 0.45 && abs($0.z) < 0.45 })
        let mean = mesh.colors.reduce(SIMD3<Float>.zero, +) / Float(mesh.colors.count)
        #expect(mean.x > mean.y)            // f_dc (1, −1, 0) → reddish
        #expect(mesh.quads.allSatisfy { max($0.x, $0.y, $0.z, $0.w) < UInt32(mesh.positions.count) })
    }

    @Test func glbIsWellFormed() throws {
        let mesh = MeshExtraction.extract(Self.ball(), options: .init(resolution: 48))
        let d = GLBWriter.data(mesh)
        func u32(_ o: Int) -> Int { Int(d.subdata(in: o..<(o + 4)).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }) }
        #expect(u32(0) == 0x4654_6C67 && u32(4) == 2 && u32(8) == d.count)
        let jsonLength = u32(12)
        let json = try JSONSerialization.jsonObject(with: d.subdata(in: 20..<(20 + jsonLength))) as! [String: Any]
        let accessors = json["accessors"] as! [[String: Any]]
        #expect(accessors[0]["count"] as! Int == mesh.positions.count)
        #expect(accessors[3]["count"] as! Int == 3 * mesh.triangleCount)
        #expect(u32(20 + jsonLength) == (json["buffers"] as! [[String: Any]])[0]["byteLength"] as! Int)
    }

    @Test func usdzIsAlignedAndChecksummed() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).usdz")
        defer { try? FileManager.default.removeItem(at: url) }
        try USDZWriter.write(MeshExtraction.extract(Self.ball(), options: .init(resolution: 32)), to: url)
        let d = try Data(contentsOf: url)
        var offset = 0, names: [String] = []
        while d.subdata(in: offset..<(offset + 4)) == Data([0x50, 0x4B, 0x03, 0x04]) {
            func u16(_ o: Int) -> Int { Int(d[offset + o]) | Int(d[offset + o + 1]) << 8 }
            func u32(_ o: Int) -> UInt32 { d.subdata(in: (offset + o)..<(offset + o + 4)).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) } }
            let nameLength = u16(26), extra = u16(28), size = Int(u32(18))
            let start = offset + 30 + nameLength + extra
            names.append(String(decoding: d.subdata(in: (offset + 30)..<(offset + 30 + nameLength)), as: UTF8.self))
            #expect(start % 64 == 0, "file data must start on a 64-byte boundary")
            #expect(USDZip.crc32(d.subdata(in: start..<(start + size))) == u32(14))
            offset = start + size
        }
        #expect(names == ["Object.usda", "texture.png"])
    }
}
