// PrintMesh.swift - a Gaussian cloud → a closed, solid, single-piece mesh in millimetres for 3D printing (STL)
// Copyright 2026 Vincent Gourbin

import Foundation
import simd

public struct PrintOptions: Sendable {
    public enum Base: Sendable, Equatable {
        /// As generated (a rounded bottom may not stand).
        case none
        /// The bottom is cut flat (`flatCutFraction` of the height) so it stands on the bed.
        case flat
        /// A slab under the object's top-down silhouette (joins legs, adds stability).
        case plinth
    }
    /// Height of the printed object (z, the bed's normal).
    public var heightMM: Float = 100
    /// Cells of the clean-up grid over the object's height (192 on 100 mm ≈ 0.5 mm). Every Gaussian is an
    /// exact kernel, widened to 0.6 cell when thinner, so a finer grid sharpens without pitting.
    public var gridCells = 192
    /// Every part ends up at least this thick (wires, sheets, ears); cracks narrower than it close.
    public var minThicknessMM: Float = 1.2
    public var base: Base = .flat
    public var flatCutFraction: Float = 0.02
    public var plinthThicknessMM: Float = 3
    public var plinthMarginMM: Float = 3
    /// Quadric decimation down to this many triangles (0: none).
    public var maxTriangles = 200_000
    /// Taubin smoothing passes: 60 removes the one-cell voxel steps without shrinking the object
    /// (foosball, 2026-11-16: volume 512.9 → 511.9 cm³, steps gone; 10 left them).
    public var smoothingIterations = 60
    /// Gaussians below this opacity are ignored. 0: a solid needs every Gaussian — the top of the fox's head is
    /// made of Gaussians below 0.05 opacity, which only add up to an opaque surface along a ray.
    public var minimumOpacity: Float = 0

    public init(heightMM: Float = 100, minThicknessMM: Float = 1.2, base: Base = .flat, maxTriangles: Int = 200_000) {
        self.heightMM = heightMM
        self.minThicknessMM = minThicknessMM
        self.base = base
        self.maxTriangles = maxTriangles
    }
}

/// What a slicer will care about, checked on the final triangles.
public struct PrintReport: Sendable, CustomStringConvertible {
    public var triangleCount: Int
    public var sizeMM: SIMD3<Float>
    public var volumeCM3: Double
    /// Separate closed surfaces (1: one piece).
    public var shells: Int
    /// Every edge has exactly two triangles, in opposite directions: no hole, consistent orientation.
    public var isWatertight: Bool
    /// Watertight and every vertex has a single fan of triangles.
    public var isManifold: Bool
    public var genus: Int
    /// Floating pieces dropped from the generated shape.
    public var removedFragments: Int
    public var isPrintable: Bool { isWatertight && isManifold && shells == 1 && volumeCM3 > 0 }

    public var description: String {
        String(format: "%d triangles, %.1f × %.1f × %.1f mm, %.1f cm³, %d shell(s), genus %d, %@, %d fragment(s) removed",
               triangleCount, sizeMM.x, sizeMM.y, sizeMM.z, volumeCM3, shells, genus,
               isPrintable ? "printable (watertight, manifold)" : "NOT printable"
                   + (isWatertight ? "" : " — not watertight") + (isManifold ? "" : " — not manifold"),
               removedFragments)
    }
}

/// A printable mesh: millimetres, z up, standing on z = 0, centred on x = y = 0.
public struct PrintMesh: Sendable {
    public var positions: [SIMD3<Float>]
    public var triangles: [SIMD3<UInt32>]
    public var report: PrintReport
}

public enum PrintPreparation {
    /// Extinction grid → solid (what is not seen from above or the sides; largest piece, minimum thickness, base,
    /// ambiguity-free) →
    /// surface nets → Taubin smoothing → quadric decimation → millimetres, checked.
    public static func prepare(_ cloud: GaussianCloud, options: PrintOptions = .init()) -> PrintMesh? {
        let opt = options
        var clock = Date()
        let timing = ProcessInfo.processInfo.environment["IDOL_PRINT_TIMING"] != nil
        func lap(_ stage: String) {
            if timing { print(String(format: "  print %@ %.2f s", stage, Date().timeIntervalSince(clock))) }
            clock = Date()
        }
        let rMM = opt.minThicknessMM / 2
        let plinthMM = opt.base == .plinth ? opt.plinthThicknessMM + opt.plinthMarginMM : 0
        // first pass: cell from the Gaussians' height; a second pass corrects it once the floaters are gone
        var cellScale: Float = 1
        var built: (solid: VoxelSolid, grid: Grid, fragments: Int, mmPerCell: Float)?
        for attempt in 0..<2 {
            let targetCells = Float(opt.gridCells)
            let padding = Int(((rMM + plinthMM) / (opt.heightMM / targetCells)).rounded(.up)) + 4
            guard let g = OpacityField.grid(around: cloud, minimumOpacity: opt.minimumOpacity,
                                            cell: { $0.z / targetCells * cellScale }, padding: padding) else { return nil }
            let kappa = OpacityField.extinction(cloud, grid: g, minimumOpacity: opt.minimumOpacity)
            lap("extinction \(g.dims)")
            var solid = OpacityField.solid(kappa: kappa, grid: g)
            lap("solid (seen from above and the sides)")
            let fragments = solid.keepLargestComponent()
            guard let zr = solid.zRange else { return nil }
            solid.fillCavities()
            lap("pieces + cavities")
            let mmPerCell = opt.heightMM / Float(zr.count)
            built = (solid, g, fragments, mmPerCell)
            let ratio = mmPerCell / (opt.heightMM / targetCells)
            if attempt == 0 && (ratio > 1.25 || ratio < 0.8) { cellScale = 1 / ratio } else { break }
        }
        guard var (solid, grid, fragments, mmPerCell) = built else { return nil }
        solid.enforceThickness(radius: rMM / mmPerCell)
        lap("thickness")
        switch opt.base {
        case .none: break
        case .flat:
            if let zr = solid.zRange { solid.cut(below: zr.lowerBound + max(1, Int((Float(zr.count) * opt.flatCutFraction).rounded()))) }
        case .plinth:
            solid.addPlinth(thickness: max(1, Int((opt.plinthThicknessMM / mmPerCell).rounded())),
                            margin: opt.plinthMarginMM / mmPerCell)
        }
        fragments += solid.keepLargestComponent()
        for _ in 0..<4 {
            solid.fillCavities()
            if solid.makeWellComposed() == 0 { break }
        }
        lap("base + well-composed")
        let smooth = grid.blur(solid.v.map { Float($0) })
        var mesh = TriangleMesh.extract(solid, smooth: smooth)
        lap("surface (\(mesh.triangles.count) triangles)")
        mesh.taubin(iterations: opt.smoothingIterations)
        lap("smoothing")
        if opt.maxTriangles > 0 { Decimation.run(&mesh, targetTriangles: opt.maxTriangles) }
        lap("decimation")
        // cells → millimetres, z up, on the bed, centred
        var lo = SIMD3<Double>(repeating: .greatestFiniteMagnitude), hi = -lo
        for p in mesh.positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        guard lo.z < hi.z else { return nil }
        let scale = Double(opt.heightMM) / (hi.z - lo.z)
        let centre = SIMD3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z)
        mesh.positions = mesh.positions.map { ($0 - centre) * scale }
        let check = mesh.check()
        let size = SIMD3<Float>((hi - lo) * scale)
        let report = PrintReport(triangleCount: mesh.triangles.count, sizeMM: size, volumeCM3: check.volume / 1000,
                                 shells: check.shells, isWatertight: check.watertight, isManifold: check.manifold,
                                 genus: check.genus, removedFragments: fragments)
        return PrintMesh(positions: mesh.positions.map { SIMD3<Float>($0) },
                         triangles: mesh.triangles.map { SIMD3<UInt32>(truncatingIfNeeded: $0) }, report: report)
    }
}

/// Binary STL (millimetres, the unit every slicer assumes).
public enum STLWriter {
    public static func write(_ mesh: PrintMesh, to url: URL) throws {
        var data = Data(count: 80)
        data.replaceSubrange(0..<min(80, 40), with: Array("sperid-swift-mlx Idol, millimetres, z up".utf8.prefix(40)))
        var n = UInt32(mesh.triangles.count).littleEndian
        withUnsafeBytes(of: &n) { data.append(contentsOf: $0) }
        data.reserveCapacity(84 + 50 * mesh.triangles.count)
        var record = [Float](repeating: 0, count: 12)
        for t in mesh.triangles {
            let a = mesh.positions[Int(t.x)], b = mesh.positions[Int(t.y)], c = mesh.positions[Int(t.z)]
            let nrm = simd_normalize(simd_cross(b - a, c - a))
            let normal = nrm.x.isFinite ? nrm : SIMD3<Float>(0, 0, 1)
            for (k, v) in [normal, a, b, c].enumerated() { record[3 * k] = v.x; record[3 * k + 1] = v.y; record[3 * k + 2] = v.z }
            record.withUnsafeBytes { data.append(contentsOf: $0) }
            data.append(contentsOf: [0, 0])
        }
        try data.write(to: url, options: .atomic)
    }
}
