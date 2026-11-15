// SurfaceNets.swift - naive surface nets on a density grid: one vertex per crossing cell, one quad per crossing edge
// Copyright 2026 Vincent Gourbin

import Foundation
import simd

enum SurfaceNets {
    static let cubeEdges: [(Int, Int)] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    static func corner(_ c: Int) -> SIMD3<Float> { SIMD3(Float(c & 1), Float((c >> 1) & 1), Float((c >> 2) & 1)) }

    static func extract(density: [Float], color: [SIMD3<Float>], grid: Grid, iso: Float) -> SplatMesh {
        let (nx, ny, nz) = (grid.dims.x, grid.dims.y, grid.dims.z)
        func f(_ x: Int, _ y: Int, _ z: Int) -> Float { density[grid.index(x, y, z)] }
        // 1. a vertex in every cell whose 8 corners straddle the iso-level, at the mean of its edge crossings
        var vertexOf = [Int32](repeating: -1, count: (nx - 1) * (ny - 1) * (nz - 1))
        func cellIndex(_ x: Int, _ y: Int, _ z: Int) -> Int { (z * (ny - 1) + y) * (nx - 1) + x }
        var mesh = SplatMesh()
        var gridPoints: [SIMD3<Float>] = []
        for z in 0..<(nz - 1) { for y in 0..<(ny - 1) { for x in 0..<(nx - 1) {
            var v = [Float](repeating: 0, count: 8), mask = 0
            for c in 0..<8 {
                v[c] = f(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1))
                if v[c] > iso { mask |= 1 << c }
            }
            guard mask != 0 && mask != 255 else { continue }
            var sum = SIMD3<Float>.zero, crossings: Float = 0
            for (a, b) in cubeEdges where (v[a] > iso) != (v[b] > iso) {
                let t = (iso - v[a]) / (v[b] - v[a])
                sum += corner(a) + (corner(b) - corner(a)) * t
                crossings += 1
            }
            vertexOf[cellIndex(x, y, z)] = Int32(gridPoints.count)
            gridPoints.append(SIMD3(Float(x), Float(y), Float(z)) + sum / crossings)
        } } }
        // 2. a quad around every grid edge that crosses the iso-level (its 4 neighbouring cells)
        for z in 1..<(nz - 1) { for y in 1..<(ny - 1) { for x in 1..<(nx - 1) {
            let inside = f(x, y, z) > iso
            for axis in 0..<3 {
                let (ex, ey, ez) = axis == 0 ? (1, 0, 0) : (axis == 1 ? (0, 1, 0) : (0, 0, 1))
                guard x + ex < nx - 1 || axis != 0, y + ey < ny - 1 || axis != 1, z + ez < nz - 1 || axis != 2 else { continue }
                guard inside != (f(x + ex, y + ey, z + ez) > iso) else { continue }
                // the 4 cells sharing this edge, in order around it
                let cells: [(Int, Int, Int)]
                switch axis {
                case 0: cells = [(x, y - 1, z - 1), (x, y, z - 1), (x, y, z), (x, y - 1, z)]
                case 1: cells = [(x - 1, y, z - 1), (x - 1, y, z), (x, y, z), (x, y, z - 1)]
                default: cells = [(x - 1, y - 1, z), (x, y - 1, z), (x, y, z), (x - 1, y, z)]
                }
                let ids = cells.map { vertexOf[cellIndex($0.0, $0.1, $0.2)] }
                guard ids.allSatisfy({ $0 >= 0 }) else { continue }
                let q = SIMD4<UInt32>(UInt32(ids[0]), UInt32(ids[1]), UInt32(ids[2]), UInt32(ids[3]))
                mesh.quads.append(inside ? q : SIMD4(q.w, q.z, q.y, q.x))
            }
        } } }
        // 3. positions, normals (−∇density), colours, in the y-up export frame
        for g in gridPoints {
            mesh.positions.append(SplatMesh.yUp(grid.position(g)))
            mesh.normals.append(SplatMesh.yUp(-gradient(density, grid, g)))
            mesh.colors.append(sample(color, grid, g))
        }
        return mesh
    }

    static func sample<T: BlurValue>(_ a: [T], _ grid: Grid, _ g: SIMD3<Float>) -> T {
        var acc = T.zero
        grid.trilinear(grid.position(g)) { i, w in acc = acc + a[i].scaled(w) }
        return acc
    }

    static func gradient(_ d: [Float], _ grid: Grid, _ g: SIMD3<Float>) -> SIMD3<Float> {
        let e: Float = 0.5
        let gx = sample(d, grid, g + SIMD3(e, 0, 0)) - sample(d, grid, g - SIMD3(e, 0, 0))
        let gy = sample(d, grid, g + SIMD3(0, e, 0)) - sample(d, grid, g - SIMD3(0, e, 0))
        let gz = sample(d, grid, g + SIMD3(0, 0, e)) - sample(d, grid, g - SIMD3(0, 0, e))
        let v = SIMD3(gx, gy, gz)
        let n = (v * v).sum().squareRoot()
        return n > 1e-12 ? v / n : SIMD3(0, 0, 1)
    }
}
