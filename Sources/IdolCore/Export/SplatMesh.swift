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
    /// Grid cells along the object's longest side.
    public var resolution = 192
    /// Unused since 0.4: the surface is where a ray loses half its light (`OpacityField`), not an iso-level of a
    /// blurred density — that one sat in the Gaussians' tails and made every object puffy.
    @available(*, deprecated, message: "the surface is the transmittance-0.5 surface; this value is ignored")
    public var isoFraction: Float {
        get { 0 }
        set { }
    }
    /// Gaussians below this opacity are ignored. 0: thin parts (the top of the fox's head) are made of Gaussians
    /// below 0.05 that only add up to an opaque surface along a ray.
    public var minimumOpacity: Float = 0
    /// Taubin passes that remove the one-cell voxel steps.
    public var smoothingIterations = 60
    /// Pieces smaller than this fraction of the largest are dropped (floaters).
    public var minimumPieceFraction = 0.01

    public init(resolution: Int = 192, minimumOpacity: Float = 0, smoothingIterations: Int = 60) {
        self.resolution = resolution
        self.minimumOpacity = minimumOpacity
        self.smoothingIterations = smoothingIterations
    }

    @available(*, deprecated, message: "isoFraction is ignored since 0.4")
    public init(resolution: Int = 192, isoFraction: Float, minimumOpacity: Float = 0) {
        self.init(resolution: resolution, minimumOpacity: minimumOpacity)
    }
}

public enum MeshExtraction {
    static let shC0: Float = 0.28209479177387814

    /// Every Gaussian as an exact extinction kernel; the surface a renderer shows (transmittance 0.5, seen from
    /// above and from the sides), floaters dropped; surface nets, Taubin smoothing; colours from the
    /// Gaussians that make the surface; y-up, about 1 m.
    public static func extract(_ cloud: GaussianCloud, options: MeshExtractionOptions = .init()) -> SplatMesh {
        let n = max(16, options.resolution)
        guard let grid = OpacityField.grid(around: cloud, minimumOpacity: options.minimumOpacity,
                                           cell: { $0.max() / Float(n - 8) }, padding: 4) else { return SplatMesh() }
        let field = OpacityField.extinction(cloud, grid: grid, minimumOpacity: options.minimumOpacity, withColor: true)
        var solid = OpacityField.solid(kappa: field.kappa, grid: grid)
        solid.keepLargestComponent(minFraction: options.minimumPieceFraction)
        for _ in 0..<4 {
            solid.fillCavities()
            if solid.makeWellComposed() == 0 { break }
        }
        var (positions, quads) = TriangleMesh.surfaceQuads(solid, smooth: grid.blur(solid.v.map { Float($0) }))
        guard !quads.isEmpty else { return SplatMesh() }
        TriangleMesh.taubin(&positions, edges: quads.flatMap { [($0.x, $0.y), ($0.y, $0.z), ($0.z, $0.w), ($0.w, $0.x)] },
                            iterations: options.smoothingIterations)
        // colours: κ-weighted, spread a little so vertices just outside the opaque layer still find them
        let colorSum = grid.blur(grid.blur(field.color)), weight = grid.blur(grid.blur(field.kappa))
        var normals = [SIMD3<Float>](repeating: .zero, count: positions.count)
        for q in quads {
            let p = [q.x, q.y, q.z, q.w].map { positions[Int($0)] }
            let nrm = SIMD3<Float>(simd_cross(p[2] - p[0], p[3] - p[1]))
            for v in [q.x, q.y, q.z, q.w] { normals[Int(v)] += nrm }
        }
        var mesh = SplatMesh()
        mesh.quads = quads.map { SIMD4<UInt32>(truncatingIfNeeded: $0) }
        mesh.positions.reserveCapacity(positions.count)
        for (i, p) in positions.enumerated() {
            let g = SIMD3<Float>(p)
            mesh.positions.append(SplatMesh.yUp(grid.position(g)))
            let len = simd_length(normals[i])
            mesh.normals.append(SplatMesh.yUp(len > 0 ? normals[i] / len : SIMD3(0, 0, 1)))
            let w = SurfaceNets.sample(weight, grid, g)
            mesh.colors.append(w > 1e-6 ? simd_clamp(SurfaceNets.sample(colorSum, grid, g) / w, .zero, SIMD3(repeating: 1))
                                        : SIMD3(repeating: 0.5))
        }
        return mesh
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
