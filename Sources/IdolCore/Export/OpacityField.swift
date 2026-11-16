// OpacityField.swift - the splat as an extinction volume, and the surface a renderer would show (transmittance 0.5)
// Copyright 2026 Vincent Gourbin

import Foundation
import simd

/// The point-sampled, blurred density of `MeshExtraction` puts the surface in the Gaussians' tails (a low iso
/// over a blurred grid): objects come out puffy, by 2-3 cells. Here every Gaussian is an exact extinction kernel,
/// and the surface is where a ray from outside has lost half its light — what a 3DGS render shows.
enum OpacityField {
    /// Extinction per unit length: Σᵢ kᵢ · exp(−½ (x − μᵢ)ᵀ Σᵢ⁻¹ (x − μᵢ)), with kᵢ such that a ray through the
    /// centre along the thinnest axis loses exactly αᵢ (the 3DGS alpha of a face-on Gaussian):
    /// kᵢ = −ln(1 − αᵢ) / (√(2π) σ_min). Axes thinner than `minSigmaCells` are widened, keeping that optical depth
    /// (a sheet thinner than a cell would otherwise alias: sampled at cell centres it pits the surface).
    static func extinction(_ cloud: GaussianCloud, grid: Grid, minimumOpacity: Float, minSigmaCells: Float = 0.6) -> [Float] {
        extinction(cloud, grid: grid, minimumOpacity: minimumOpacity, minSigmaCells: minSigmaCells, withColor: false).kappa
    }

    /// … and, with `withColor`, Σ κᵢ·rgbᵢ (sRGB from the DC coefficient), for vertex colours.
    static func extinction(_ cloud: GaussianCloud, grid: Grid, minimumOpacity: Float, minSigmaCells: Float = 0.6,
                           withColor: Bool) -> (kappa: [Float], color: [SIMD3<Float>]) {
        let (nx, ny, nz) = (grid.dims.x, grid.dims.y, grid.dims.z)
        var kappa = [Float](repeating: 0, count: grid.count)
        var color = [SIMD3<Float>](repeating: .zero, count: withColor ? grid.count : 0)
        struct Kernel { var mu: SIMD3<Float>; var p: simd_float3x3; var k: Float; var z0: Int; var z1: Int; var ext: SIMD3<Float>; var rgb: SIMD3<Float> }
        var kernels: [Kernel] = []
        kernels.reserveCapacity(cloud.count)
        let sqrt2pi = Float(2 * Double.pi).squareRoot()
        for i in 0..<cloud.count {
            let alpha = min(cloud.opacity[i], 0.995)
            guard alpha >= max(minimumOpacity, 1e-4) else { continue }
            let mu = (SIMD3(cloud.xyz[3 * i], cloud.xyz[3 * i + 1], cloud.xyz[3 * i + 2]) - grid.origin) / grid.cell
            let q = simd_quatf(ix: cloud.rotation[4 * i + 1], iy: cloud.rotation[4 * i + 2], iz: cloud.rotation[4 * i + 3],
                               r: cloud.rotation[4 * i])
            let r = simd_float3x3(q.normalized)
            let s = simd_max(SIMD3(cloud.scaling[3 * i], cloud.scaling[3 * i + 1], cloud.scaling[3 * i + 2]) / grid.cell,
                             SIMD3(repeating: minSigmaCells))
            let inv = simd_float3x3(diagonal: 1 / (s * s))
            let p = r * inv * r.transpose                       // precision, in cells
            let cov = r * simd_float3x3(diagonal: s * s) * r.transpose
            let ext = 3 * SIMD3(cov[0][0].squareRoot(), cov[1][1].squareRoot(), cov[2][2].squareRoot())   // 3σ AABB
            let z0 = max(0, Int((mu.z - ext.z).rounded(.up))), z1 = min(nz - 1, Int((mu.z + ext.z).rounded(.down)))
            guard z0 <= z1 else { continue }
            let rgb = simd_clamp(0.5 + MeshExtraction.shC0 * SIMD3(cloud.featuresDC[3 * i], cloud.featuresDC[3 * i + 1], cloud.featuresDC[3 * i + 2]),
                                 SIMD3(repeating: 0), SIMD3(repeating: 1))
            kernels.append(Kernel(mu: mu, p: p, k: -log(1 - alpha) / (sqrt2pi * s.min()) * 1 / grid.cell, z0: z0, z1: z1, ext: ext, rgb: rgb))
        }
        // z slabs, one per task: each writes only its own rows
        let slabs = min(nz, 64), slabHeight = (nz + slabs - 1) / slabs
        var buckets = [[Int32]](repeating: [], count: slabs)
        for (i, g) in kernels.enumerated() {
            for b in (g.z0 / slabHeight)...(g.z1 / slabHeight) { buckets[b].append(Int32(i)) }
        }
        do {
            let ks = kernels
            color.withUnsafeMutableBufferPointer { cbuf in
            // an empty array still has a base address (the shared empty storage, read-only): never write it
            let colorOut = withColor ? cbuf.baseAddress : nil
            kappa.withUnsafeMutableBufferPointer { buf in
                let out = buf.baseAddress!
                DispatchQueue.concurrentPerform(iterations: slabs) { b in
                    let zs = b * slabHeight, ze = min(nz - 1, zs + slabHeight - 1)
                    guard zs <= ze else { return }
                    for gi in buckets[b] {
                        let g = ks[Int(gi)]
                        let p = g.p
                        let (pxx, pxy, pxz, pyy, pyz, pzz) = (p[0][0], p[0][1], p[0][2], p[1][1], p[1][2], p[2][2])
                        let y0 = max(0, Int((g.mu.y - g.ext.y).rounded(.up))), y1 = min(ny - 1, Int((g.mu.y + g.ext.y).rounded(.down)))
                        guard y0 <= y1 else { continue }
                        for z in max(zs, g.z0)...min(ze, g.z1) {
                            let dz = Float(z) - g.mu.z
                            for y in y0...y1 {
                                let dy = Float(y) - g.mu.y
                                // q(dx) = pxx dx² + 2 dx (pxy dy + pxz dz) + c ≤ 9
                                let bq = pxy * dy + pxz * dz
                                let c = pyy * dy * dy + 2 * pyz * dy * dz + pzz * dz * dz
                                let disc = bq * bq - pxx * (c - 9)
                                guard disc > 0 else { continue }
                                let root = disc.squareRoot()
                                let xa = g.mu.x + (-bq - root) / pxx, xb = g.mu.x + (-bq + root) / pxx
                                let x0 = max(0, Int(xa.rounded(.up))), x1 = min(nx - 1, Int(xb.rounded(.down)))
                                guard x0 <= x1 else { continue }
                                let row = (z * ny + y) * nx
                                for x in x0...x1 {
                                    let dx = Float(x) - g.mu.x
                                    let qv = pxx * dx * dx + 2 * dx * bq + c
                                    let w = g.k * exp(-0.5 * qv)
                                    out[row + x] += w
                                    if let colorOut { colorOut[row + x] += w * g.rgb }
                                }
                            }
                        }
                    }
                }
            }
            }
        }
        return (kappa, color)
    }

    /// A grid around the Gaussians kept (centres), `cell` from their extent, `padding` empty cells around.
    static func grid(around cloud: GaussianCloud, minimumOpacity: Float, cell cellFor: (SIMD3<Float>) -> Float,
                     padding: Int) -> Grid? {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        for i in 0..<cloud.count where cloud.opacity[i] >= max(minimumOpacity, 1e-4) {
            let p = SIMD3(cloud.xyz[3 * i], cloud.xyz[3 * i + 1], cloud.xyz[3 * i + 2])
            lo = simd_min(lo, p); hi = simd_max(hi, p)
        }
        guard lo.x < hi.x else { return nil }
        let cell = cellFor(hi - lo)
        lo -= Float(padding) * cell
        let dims = SIMD3<Int>(Int(((hi.x - lo.x) / cell).rounded(.up)) + padding + 1,
                              Int(((hi.y - lo.y) / cell).rounded(.up)) + padding + 1,
                              Int(((hi.z - lo.z) / cell).rounded(.up)) + padding + 1)
        return Grid(dims: dims, origin: lo, cell: cell)
    }

    /// The 26 directions of the 3×3×3 neighbourhood.
    static let directions: [SIMD3<Int>] = {
        var d: [SIMD3<Int>] = []
        for z in -1...1 { for y in -1...1 { for x in -1...1 where !(x == 0 && y == 0 && z == 0) { d.append(SIMD3(x, y, z)) } } }
        return d
    }()

    /// The object as a solid: every voxel a viewer **above or level with it** can see (a ray from outside,
    /// travelling down or horizontally along one of the 17 such directions of the 3×3×3 neighbourhood, reaches it
    /// with more than `surface` of its light) is empty; the rest is matter. The surface therefore sits where a
    /// render shows it (transmittance 0.5). Rays from below are not used: the model never sees the underside,
    /// which is often open, and through it they would empty the inside. Open tops (a foosball's pitch, a cup)
    /// stay open.
    static func solid(kappa: [Float], grid: Grid, surface: Float = 0.5) -> VoxelSolid {
        let (nx, ny, nz) = (grid.dims.x, grid.dims.y, grid.dims.z)
        var seen = [UInt8](repeating: 0, count: grid.count)
        let tauSurface = -log(surface)
        func inBounds(_ p: SIMD3<Int>) -> Bool { p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < nx && p.y < ny && p.z < nz }
        var boundary: [SIMD3<Int>] = []
        for z in 0..<nz { for y in 0..<ny { for x in 0..<nx where x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1 {
            boundary.append(SIMD3(x, y, z))
        } } }
        seen.withUnsafeMutableBufferPointer { sbuf in
            let out = sbuf.baseAddress!
            kappa.withUnsafeBufferPointer { kbuf in
                let k = kbuf.baseAddress!
                for d in directions where d.z <= 0 {
                    let step = grid.cell * simd_length(SIMD3<Float>(Float(d.x), Float(d.y), Float(d.z)))
                    let starts = boundary.filter { !inBounds($0 &- d) }
                    // lines of one direction never share a voxel: no race
                    DispatchQueue.concurrentPerform(iterations: starts.count / 256 + 1) { chunk in
                        let a = chunk * 256, e = min(starts.count, a + 256)
                        guard a < e else { return }
                        for s in starts[a..<e] {
                            var p = s, tau: Float = 0
                            while inBounds(p) {
                                let i = (p.z * ny + p.y) * nx + p.x
                                tau += k[i] * step
                                if tau >= tauSurface { break }
                                out[i] = 1
                                p &+= d
                            }
                        }
                    }
                }
            }
        }
        return VoxelSolid(nx: nx, ny: ny, nz: nz, v: seen.map { 1 - $0 })
    }
}
