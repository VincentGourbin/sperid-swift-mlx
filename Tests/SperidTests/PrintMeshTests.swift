// PrintMeshTests.swift - the 3D-print clean-up on synthetic solids: closed, manifold, right genus and volume (tier 0)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
import simd
@testable import IdolCore

private func solid(_ n: Int, _ inside: (SIMD3<Double>) -> Bool) -> VoxelSolid {
    var s = VoxelSolid(nx: n, ny: n, nz: n)
    for z in 1..<(n - 1) { for y in 1..<(n - 1) { for x in 1..<(n - 1) where inside(SIMD3(Double(x), Double(y), Double(z))) {
        s[x, y, z] = true
    } } }
    return s
}

private func mesh(_ s: VoxelSolid) -> TriangleMesh {
    var w = s
    w.makeWellComposed()
    let grid = Grid(dims: SIMD3(w.nx, w.ny, w.nz), origin: .zero, cell: 1)
    return TriangleMesh.extract(w, smooth: grid.blur(w.v.map { Float($0) }))
}

@Suite("PrintMesh")
struct PrintMeshTests {
    let c = SIMD3<Double>(repeating: 24)

    @Test func sphereIsClosedGenusZeroWithItsVolume() {
        let r = 15.0
        let m = mesh(solid(48) { simd_length($0 - c) <= r })
        let k = m.check()
        #expect(k.watertight && k.manifold)
        #expect(k.shells == 1 && k.genus == 0)
        let exact = 4.0 / 3.0 * .pi * r * r * r
        #expect(abs(k.volume - exact) / exact < 0.05, "volume \(k.volume) vs \(exact)")
    }

    @Test func torusHasGenusOne() {
        let m = mesh(solid(48) { p in
            let q = p - c
            let ring = (q.x * q.x + q.y * q.y).squareRoot() - 13
            return ring * ring + q.z * q.z <= 25
        })
        let k = m.check()
        #expect(k.watertight && k.manifold && k.shells == 1 && k.genus == 1)
    }

    /// Voxels touching only along an edge or at a corner make a naive surface non-manifold; random noise has
    /// every such configuration. Matter keeps a margin from the grid border, as in `PrintPreparation` (the
    /// repair never writes the border).
    @Test(arguments: [30, 45, 60])
    func ambiguousConfigurationsAndNoiseStayManifold(percent: Int) {
        var edge = VoxelSolid(nx: 10, ny: 10, nz: 10)
        edge[4, 4, 4] = true; edge[5, 5, 4] = true       // edge contact
        edge[4, 4, 6] = true; edge[5, 5, 7] = true       // corner contact
        var cases = [("edge+corner", edge)]
        var rng = SystemRandomNumberGenerator()
        for k in 0..<20 {
            var noise = VoxelSolid(nx: 28, ny: 28, nz: 28)
            for z in 4..<24 { for y in 4..<24 { for x in 4..<24 where Int.random(in: 0..<100, using: &rng) < percent { noise[x, y, z] = true } } }
            cases.append(("noise \(percent)% #\(k)", noise))
        }
        for (name, s) in cases {
            let k = mesh(s).check()
            #expect(k.watertight && k.manifold, "\(name): watertight \(k.watertight) manifold \(k.manifold)")
            #expect(k.volume > 0, "\(name)")   // triangles face outwards
        }
    }

    @Test func decimationKeepsAPrintableShape() {
        var m = mesh(solid(48) { simd_length($0 - c) <= 18 })
        let before = m.check().volume
        let target = m.triangles.count / 5
        Decimation.run(&m, targetTriangles: target)
        let k = m.check()
        #expect(m.triangles.count <= target + 2)
        #expect(k.watertight && k.manifold && k.genus == 0)
        #expect(abs(k.volume - before) / before < 0.02)
    }

    /// Extinction of an opaque shape (κ · cell = 3 inside: a cell absorbs 95 %).
    private func kappa(_ n: Int, _ inside: (SIMD3<Double>) -> Bool) -> (grid: Grid, kappa: [Float]) {
        let grid = Grid(dims: SIMD3(n, n, n), origin: .zero, cell: 1)
        var k = [Float](repeating: 0, count: grid.count)
        for z in 0..<n { for y in 0..<n { for x in 0..<n where inside(SIMD3(Double(x), Double(y), Double(z))) {
            k[grid.index(x, y, z)] = 3
        } } }
        return (grid, k)
    }

    /// A shell open at the bottom (the side the model never sees) comes out solid: only rays from above and the
    /// sides empty space.
    @Test func shellOpenBelowIsSolid() {
        let f = kappa(48) { p in let d = simd_length(p - self.c); return d <= 15 && d >= 13 && p.z > 20 }
        let s = OpacityField.solid(kappa: f.kappa, grid: f.grid)
        #expect(s[24, 24, 24])                       // centre
        #expect(s[24, 24, 22])                       // just above the opening
        #expect(!s[24, 24, 41])                      // above the top
    }

    /// An open-top box (a foosball's pitch, a cup) stays hollow; the gap between two legs stays empty.
    @Test func openTopAndGapsStayEmpty() {
        let box = kappa(40) { p in
            let inBox = p.x >= 8 && p.x <= 32 && p.y >= 8 && p.y <= 32 && p.z >= 8 && p.z <= 24
            let cavity = p.x > 10 && p.x < 30 && p.y > 10 && p.y < 30 && p.z > 10
            return inBox && !cavity
        }
        let s = OpacityField.solid(kappa: box.kappa, grid: box.grid)
        #expect(s[20, 20, 9])                        // the floor
        #expect(!s[20, 20, 18])                      // inside the box: seen from above
        let legs = kappa(32) { p in (p.z < 20 && (abs(p.x - 10) < 2 || abs(p.x - 22) < 2) && abs(p.y - 16) < 2) || (p.z >= 20 && p.z < 24 && p.x > 8 && p.x < 24 && abs(p.y - 16) < 2) }
        #expect(!OpacityField.solid(kappa: legs.kappa, grid: legs.grid)[16, 16, 10])
    }

    @Test func thinPartsGetTheMinimumThickness() {
        var plate = solid(32) { p in p.z == 16 && p.x > 6 && p.x < 26 && p.y > 6 && p.y < 26 }   // one cell thick
        plate.enforceThickness(radius: 1.5)
        let column = (0..<32).filter { plate[16, 16, $0] }.count
        #expect(column >= 3, "plate is \(column) cells thick")
    }

    @Test func stlHasTheBinaryLayout() throws {
        let m = mesh(solid(24) { simd_length($0 - SIMD3(repeating: 12)) <= 8 })
        let pm = PrintMesh(positions: m.positions.map { SIMD3<Float>($0) },
                           triangles: m.triangles.map { SIMD3<UInt32>(truncatingIfNeeded: $0) },
                           report: PrintReport(triangleCount: m.triangles.count, sizeMM: .zero, volumeCM3: 0, shells: 1,
                                               isWatertight: true, isManifold: true, genus: 0, removedFragments: 0))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("idol-\(UUID()).stl")
        defer { try? FileManager.default.removeItem(at: url) }
        try STLWriter.write(pm, to: url)
        let data = try Data(contentsOf: url)
        #expect(data.count == 84 + 50 * m.triangles.count)
        let count = data.subdata(in: 80..<84).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        #expect(Int(count) == m.triangles.count)
    }
}
