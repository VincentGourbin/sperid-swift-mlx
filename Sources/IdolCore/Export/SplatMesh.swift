// SplatMesh.swift - a coloured triangle mesh extracted from a Gaussian cloud (density grid + surface nets)
// Copyright 2026 Vincent Gourbin

import Foundation
import simd

/// A quad mesh in the export frame (**y-up**, metres: the object spans about 1 m), with per-vertex normals
/// and sRGB colours. Quads are split into two triangles by the writers.
public struct SplatMesh: Sendable {
    public var positions: [SIMD3<Float>] = []
    public var normals: [SIMD3<Float>] = []
    public var colors: [SIMD3<Float>] = []         // sRGB, 0…1
    public var quads: [SIMD4<UInt32>] = []

    public var triangleCount: Int { 2 * quads.count }

    /// Object frame (z-up, `GaussianCloud`) → export frame (y-up): (x, y, z) → (x, z, −y).
    static func yUp(_ p: SIMD3<Float>) -> SIMD3<Float> { SIMD3(p.x, p.z, -p.y) }
}

public struct MeshExtractionOptions: Sendable {
    /// Grid cells per axis over the object's bounding box.
    public var resolution = 192
    /// Iso-level as a fraction of the 99th percentile of the occupied cells' density.
    /// 0.04: thin dark parts (the foosball's legs) stay closed; 0.02 lets inner Gaussians streak the
    /// surface, 0.08-0.12 opened holes (compared on renders, 2026-11-15).
    public var isoFraction: Float = 0.04
    /// Gaussians below this opacity are ignored (floaters).
    public var minimumOpacity: Float = 0.05
    public init(resolution: Int = 192, isoFraction: Float = 0.04, minimumOpacity: Float = 0.05) {
        self.resolution = resolution
        self.isoFraction = isoFraction
        self.minimumOpacity = minimumOpacity
    }
}

public enum MeshExtraction {
    static let shC0: Float = 0.28209479177387814

    /// Splats each Gaussian's opacity (and colour) trilinearly at its centre and at ±1σ along its three
    /// rotated axes (7 points: an elongated Gaussian covers its length — thin parts no longer break into
    /// beads), blurs the grid (σ ≈ 1 cell),
    /// extracts the iso-surface with surface nets, and colours / orients the vertices from the grids.
    public static func extract(_ cloud: GaussianCloud, options: MeshExtractionOptions = .init()) -> SplatMesh {
        let n = options.resolution
        // bounds of the kept Gaussians, padded by two cells
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        for i in 0..<cloud.count where cloud.opacity[i] >= options.minimumOpacity {
            let p = SIMD3(cloud.xyz[3 * i], cloud.xyz[3 * i + 1], cloud.xyz[3 * i + 2])
            lo = simd_min(lo, p); hi = simd_max(hi, p)
        }
        guard lo.x <= hi.x else { return SplatMesh() }
        let cell = (hi - lo).max() / Float(n - 4)
        lo -= 2 * cell
        let dims = SIMD3<Int>(Int(((hi.x - lo.x) / cell).rounded(.up)) + 3, Int(((hi.y - lo.y) / cell).rounded(.up)) + 3,
                              Int(((hi.z - lo.z) / cell).rounded(.up)) + 3)
        let grid = Grid(dims: dims, origin: lo, cell: cell)
        var density = [Float](repeating: 0, count: grid.count)
        var color = [SIMD3<Float>](repeating: .zero, count: grid.count)
        for i in 0..<cloud.count {
            let o = cloud.opacity[i]
            guard o >= options.minimumOpacity else { continue }
            let p = SIMD3(cloud.xyz[3 * i], cloud.xyz[3 * i + 1], cloud.xyz[3 * i + 2])
            let rgb = simd_clamp(0.5 + shC0 * SIMD3(cloud.featuresDC[3 * i], cloud.featuresDC[3 * i + 1], cloud.featuresDC[3 * i + 2]),
                                 SIMD3(repeating: 0), SIMD3(repeating: 1))
            // rotation (w, x, y, z) → the three principal axes, scaled by the Gaussian's σ
            let q = simd_quatf(ix: cloud.rotation[4 * i + 1], iy: cloud.rotation[4 * i + 2], iz: cloud.rotation[4 * i + 3],
                               r: cloud.rotation[4 * i])
            let s = SIMD3(cloud.scaling[3 * i], cloud.scaling[3 * i + 1], cloud.scaling[3 * i + 2])
            let axes = [q.act(SIMD3(s.x, 0, 0)), q.act(SIMD3(0, s.y, 0)), q.act(SIMD3(0, 0, s.z))]
            var points = [(p, Float(0.4))]
            for a in axes { points.append((p + a, 0.1)); points.append((p - a, 0.1)) }
            for (pt, share) in points {
                grid.trilinear(pt) { idx, w in density[idx] += o * w * share; color[idx] += o * w * share * rgb }
            }
        }
        density = grid.blur(density)
        color = grid.blur(color)
        for i in 0..<grid.count where density[i] > 1e-6 { color[i] /= density[i] }
        let occupied = density.filter { $0 > 1e-4 }.sorted()
        guard !occupied.isEmpty else { return SplatMesh() }
        let iso = occupied[min(occupied.count - 1, Int(Double(occupied.count) * 0.99))] * options.isoFraction
        return SurfaceNets.extract(density: density, color: color, grid: grid, iso: iso)
    }
}

struct Grid {
    let dims: SIMD3<Int>
    let origin: SIMD3<Float>
    let cell: Float
    var count: Int { dims.x * dims.y * dims.z }

    func index(_ x: Int, _ y: Int, _ z: Int) -> Int { (z * dims.y + y) * dims.x + x }

    /// World position of grid sample (x, y, z) (samples sit at cell corners of the dual grid).
    func position(_ v: SIMD3<Float>) -> SIMD3<Float> { origin + v * cell }

    func trilinear(_ p: SIMD3<Float>, _ body: (Int, Float) -> Void) {
        let g = (p - origin) / cell
        let b = SIMD3<Int>(Int(g.x.rounded(.down)), Int(g.y.rounded(.down)), Int(g.z.rounded(.down)))
        let f = g - SIMD3<Float>(Float(b.x), Float(b.y), Float(b.z))
        for dz in 0...1 { for dy in 0...1 { for dx in 0...1 {
            let x = b.x + dx, y = b.y + dy, z = b.z + dz
            guard x >= 0, y >= 0, z >= 0, x < dims.x, y < dims.y, z < dims.z else { continue }
            let w = (dx == 0 ? 1 - f.x : f.x) * (dy == 0 ? 1 - f.y : f.y) * (dz == 0 ? 1 - f.z : f.z)
            body(index(x, y, z), w)
        } } }
    }

    /// Separable [1, 4, 6, 4, 1]/16 blur (≈ Gaussian, σ ≈ 1 cell) along x, y, z; slices in parallel.
    func blur<T: BlurValue>(_ a: [T]) -> [T] {
        let k: [Float] = [1, 4, 6, 4, 1].map { $0 / 16 }
        var src = a, dst = a
        let (nx, ny, nz) = (dims.x, dims.y, dims.z)
        for axis in 0..<3 {
            let stride = axis == 0 ? 1 : (axis == 1 ? nx : nx * ny)
            let len = axis == 0 ? nx : (axis == 1 ? ny : nz)
            src.withUnsafeBufferPointer { inp in
                dst.withUnsafeMutableBufferPointer { outBuf in
                    let out = outBuf.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: nz) { z in
                        for y in 0..<ny { for x in 0..<nx {
                            let i = (z * ny + y) * nx + x
                            let c = axis == 0 ? x : (axis == 1 ? y : z)
                            var acc = T.zero
                            for t in -2...2 where c + t >= 0 && c + t < len { acc = acc + inp[i + t * stride].scaled(k[t + 2]) }
                            out[i] = acc
                        } }
                    }
                }
            }
            swap(&src, &dst)
        }
        return src
    }
}

protocol BlurValue { static var zero: Self { get }; static func + (a: Self, b: Self) -> Self; func scaled(_ s: Float) -> Self }
extension Float: BlurValue { func scaled(_ s: Float) -> Float { self * s } }
extension SIMD3: BlurValue where Scalar == Float { func scaled(_ s: Float) -> SIMD3<Float> { self * s } }
