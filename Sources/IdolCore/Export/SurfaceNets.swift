// SurfaceNets.swift - cube tables and grid sampling shared by the surface extraction (`TriangleMesh.surfaceQuads`)
// Copyright 2026 Vincent Gourbin

import Foundation
import simd

enum SurfaceNets {
    static let cubeEdges: [(Int, Int)] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    static func corner(_ c: Int) -> SIMD3<Float> { SIMD3(Float(c & 1), Float((c >> 1) & 1), Float((c >> 2) & 1)) }

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
