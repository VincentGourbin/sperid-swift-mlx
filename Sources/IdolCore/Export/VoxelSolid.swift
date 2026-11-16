// VoxelSolid.swift - a binary volume and the clean-up a 3D print needs (fragments, cavities, thickness, base)
// Copyright 2026 Vincent Gourbin

import Foundation

/// A binary volume (1 = matter) indexed like `Grid`: `(z * ny + y) * nx + x`. The operations assume (and keep)
/// an empty border, so the surface extracted from it is closed.
struct VoxelSolid {
    let nx: Int, ny: Int, nz: Int
    var v: [UInt8]

    init(nx: Int, ny: Int, nz: Int, v: [UInt8]? = nil) {
        self.nx = nx; self.ny = ny; self.nz = nz
        self.v = v ?? [UInt8](repeating: 0, count: nx * ny * nz)
    }

    var count: Int { nx * ny * nz }
    @inline(__always) func index(_ x: Int, _ y: Int, _ z: Int) -> Int { (z * ny + y) * nx + x }
    subscript(x: Int, y: Int, z: Int) -> Bool {
        get { v[index(x, y, z)] != 0 }
        set { v[index(x, y, z)] = newValue ? 1 : 0 }
    }
    var volume: Int { v.reduce(0) { $0 + Int($1) } }

    @inline(__always) func forEachNeighbour6(_ i: Int, _ body: (Int) -> Void) {
        let x = i % nx, y = (i / nx) % ny, z = i / (nx * ny)
        if x > 0 { body(i - 1) }; if x < nx - 1 { body(i + 1) }
        if y > 0 { body(i - nx) }; if y < ny - 1 { body(i + nx) }
        if z > 0 { body(i - nx * ny) }; if z < nz - 1 { body(i + nx * ny) }
    }

    // MARK: - Pieces

    /// Keeps the largest face-connected piece of matter; returns how many smaller pieces were dropped.
    @discardableResult
    mutating func keepLargestComponent() -> Int {
        var label = [Int32](repeating: 0, count: count)
        var sizes: [Int] = [0]
        var stack: [Int] = []
        for start in 0..<count where v[start] != 0 && label[start] == 0 {
            let id = Int32(sizes.count)
            var size = 0
            label[start] = id; stack.append(start)
            while let i = stack.popLast() {
                size += 1
                forEachNeighbour6(i) { j in
                    if v[j] != 0 && label[j] == 0 { label[j] = id; stack.append(j) }
                }
            }
            sizes.append(size)
        }
        guard sizes.count > 2 else { return 0 }
        let best = Int32(sizes.indices.dropFirst().max { sizes[$0] < sizes[$1] }!)
        for i in 0..<count where v[i] != 0 && label[i] != best { v[i] = 0 }
        return sizes.count - 2
    }

    /// Fills every empty region the outside cannot reach (face-connected): the object becomes solid.
    mutating func fillCavities() {
        var outside = [UInt8](repeating: 0, count: count)
        var stack: [Int] = []
        func seed(_ i: Int) { if v[i] == 0 && outside[i] == 0 { outside[i] = 1; stack.append(i) } }
        for z in 0..<nz { for y in 0..<ny {
            if z == 0 || z == nz - 1 || y == 0 || y == ny - 1 { for x in 0..<nx { seed(index(x, y, z)) } }
            else { seed(index(0, y, z)); seed(index(nx - 1, y, z)) }
        } }
        while let i = stack.popLast() {
            forEachNeighbour6(i) { j in
                if v[j] == 0 && outside[j] == 0 { outside[j] = 1; stack.append(j) }
            }
        }
        for i in 0..<count where outside[i] == 0 { v[i] = 1 }
    }

    /// The Gaussians sit on the object's surface, so the thresholded density is a shell, open where the model
    /// saw nothing (often the underside) or where it is sparse: flooding from outside would empty it. A voxel is
    /// inside when at least `votes` of the 6 axis rays from it meet matter. With 5, one open side (a bottom, a
    /// bowl-like top) no longer empties the object, while the gap between two legs (open front, back and
    /// below) stays empty.
    mutating func fillInterior(votes: Int = 5) {
        var hits = [UInt8](repeating: 0, count: count)
        let (nx, ny, nz) = (self.nx, self.ny, self.nz)
        // one line per (axis, line start): a forward and a backward scan, "matter seen so far"
        func scan(_ len: Int, _ step: Int, _ base: Int) {
            var seen = false
            for k in 0..<len { let i = base + k * step; if v[i] != 0 { seen = true } else if seen { hits[i] += 1 } }
            seen = false
            for k in stride(from: len - 1, through: 0, by: -1) {
                let i = base + k * step; if v[i] != 0 { seen = true } else if seen { hits[i] += 1 }
            }
        }
        for z in 0..<nz { for y in 0..<ny { scan(nx, 1, (z * ny + y) * nx) } }
        for z in 0..<nz { for x in 0..<nx { scan(ny, nx, z * nx * ny + x) } }
        for y in 0..<ny { for x in 0..<nx { scan(nz, nx * ny, y * nx + x) } }
        for i in 0..<count where v[i] == 0 && hits[i] >= votes { v[i] = 1 }
        fillCavities()
    }

    /// Lowest and highest z holding matter.
    var zRange: ClosedRange<Int>? {
        var lo = Int.max, hi = Int.min
        let slice = nx * ny
        for z in 0..<nz where v[(z * slice)..<((z + 1) * slice)].contains(where: { $0 != 0 }) {
            lo = min(lo, z); hi = max(hi, z)
        }
        return lo <= hi ? lo...hi : nil
    }

    // MARK: - Morphology (exact Euclidean distance, Felzenszwalb-Huttenlocher)

    /// Squared distance, in cells, from every voxel to the nearest voxel whose value is `value`.
    func squaredDistance(to value: UInt8) -> [Float] {
        var d = v.map { $0 == value ? Float(0) : Float(1e20) }
        for axis in 0..<3 { transform(&d, axis: axis) }
        return d
    }

    /// Matter grown by `r` cells (a ball).
    func dilated(_ r: Float) -> VoxelSolid {
        let d = squaredDistance(to: 1), r2 = r * r
        return VoxelSolid(nx: nx, ny: ny, nz: nz, v: d.map { $0 <= r2 ? 1 : 0 })
    }

    /// Matter shrunk by `r` cells (a ball).
    func eroded(_ r: Float) -> VoxelSolid {
        let d = squaredDistance(to: 0), r2 = r * r
        return VoxelSolid(nx: nx, ny: ny, nz: nz, v: d.map { $0 > r2 ? 1 : 0 })
    }

    /// Closes cracks narrower than `2r`, then grows every part thinner than `2r` (what an opening by a ball of
    /// radius `r` removes) by `r`: a wire or a sheet ends up at least `2r` thick, thick parts keep their shape.
    mutating func enforceThickness(radius r: Float) {
        guard r >= 0.5 else { return }
        var closed = dilated(r).eroded(r)
        for i in 0..<count where v[i] != 0 { closed.v[i] = 1 }
        let opened = closed.eroded(r).dilated(r)
        var thin = closed
        for i in 0..<count { thin.v[i] = closed.v[i] != 0 && opened.v[i] == 0 ? 1 : 0 }
        let grown = thin.dilated(r)
        for i in 0..<count { v[i] = closed.v[i] != 0 || grown.v[i] != 0 ? 1 : 0 }
    }

    private func transform(_ d: inout [Float], axis: Int) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz)
        let len = [nx, ny, nz][axis], stride = [1, nx, nx * ny][axis]
        let (outer, inner) = axis == 0 ? (nz, ny) : (axis == 1 ? (nz, nx) : (ny, nx))
        guard len > 1 else { return }
        d.withUnsafeMutableBufferPointer { buf in
            let p = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: outer) { o in
                var f = [Float](repeating: 0, count: len), out = f
                var vv = [Int](repeating: 0, count: len), zz = [Float](repeating: 0, count: len + 1)
                for k in 0..<inner {
                    let base = axis == 0 ? (o * ny + k) * nx : (axis == 1 ? o * nx * ny + k : o * nx + k)
                    for i in 0..<len { f[i] = p[base + i * stride] }
                    Self.lowerEnvelope(f, &out, &vv, &zz)
                    for i in 0..<len { p[base + i * stride] = out[i] }
                }
            }
        }
    }

    /// 1D squared distance transform: d[q] = min_j (q − j)² + f[j].
    static func lowerEnvelope(_ f: [Float], _ d: inout [Float], _ v: inout [Int], _ z: inout [Float]) {
        let n = f.count
        func s(_ q: Int, _ r: Int) -> Float {
            ((f[q] + Float(q * q)) - (f[r] + Float(r * r))) / Float(2 * q - 2 * r)
        }
        var k = 0
        v[0] = 0; z[0] = -.infinity; z[1] = .infinity
        for q in 1..<n {
            var sq = s(q, v[k])
            while sq <= z[k] { k -= 1; sq = s(q, v[k]) }
            k += 1; v[k] = q; z[k] = sq; z[k + 1] = .infinity
        }
        k = 0
        for q in 0..<n {
            while z[k + 1] < Float(q) { k += 1 }
            let dq = Float(q - v[k])
            d[q] = dq * dq + f[v[k]]
        }
    }

    // MARK: - Base

    /// Removes the matter below `z`: a flat face to stand on.
    mutating func cut(below z: Int) {
        let slice = nx * ny
        for i in 0..<(max(0, min(z, nz)) * slice) { v[i] = 0 }
    }

    /// A slab of `thickness` cells under the lowest matter: the convex hull of what touches the bed (the lowest
    /// `footFraction` of the height — legs, feet), grown by `margin` cells. It overlaps the lowest layer so the
    /// two are one piece; parts that stick out higher up (handles) leave no mark on it.
    mutating func addPlinth(thickness: Int, margin: Float, footFraction: Float = 0.1) {
        guard let zr = zRange, thickness > 0 else { return }
        let top = zr.lowerBound + max(1, Int(Float(zr.count) * footFraction))
        var points: [SIMD2<Int>] = []
        for z in zr.lowerBound..<min(top, zr.upperBound + 1) { for y in 0..<ny { for x in 0..<nx where v[index(x, y, z)] != 0 {
            points.append(SIMD2(x, y))
        } } }
        let hull = Self.convexHull(points)
        guard hull.count >= 3 else { return }
        var footprint = VoxelSolid(nx: nx, ny: ny, nz: 1)
        for y in 0..<ny { for x in 0..<nx where Self.inside(SIMD2(x, y), hull) { footprint.v[y * nx + x] = 1 } }
        if margin > 0 { footprint = footprint.dilated(margin) }
        let bottom = max(1, zr.lowerBound - thickness)
        for z in bottom...zr.lowerBound { for y in 1..<(ny - 1) { for x in 1..<(nx - 1) where footprint.v[y * nx + x] != 0 {
            v[index(x, y, z)] = 1
        } } }
    }

    /// Andrew's monotone chain, counter-clockwise.
    static func convexHull(_ points: [SIMD2<Int>]) -> [SIMD2<Int>] {
        let p = Array(Set(points)).sorted { $0.x != $1.x ? $0.x < $1.x : $0.y < $1.y }
        guard p.count >= 3 else { return p }
        func cross(_ o: SIMD2<Int>, _ a: SIMD2<Int>, _ b: SIMD2<Int>) -> Int { (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x) }
        var lower: [SIMD2<Int>] = [], upper: [SIMD2<Int>] = []
        for q in p { while lower.count >= 2 && cross(lower[lower.count - 2], lower[lower.count - 1], q) <= 0 { lower.removeLast() }; lower.append(q) }
        for q in p.reversed() { while upper.count >= 2 && cross(upper[upper.count - 2], upper[upper.count - 1], q) <= 0 { upper.removeLast() }; upper.append(q) }
        return Array(lower.dropLast() + upper.dropLast())
    }

    static func inside(_ q: SIMD2<Int>, _ hull: [SIMD2<Int>]) -> Bool {
        for i in hull.indices {
            let a = hull[i], b = hull[(i + 1) % hull.count]
            if (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x) < 0 { return false }
        }
        return true
    }

    // MARK: - Well-composedness (manifold surface)

    /// Masks of a 2×2×2 block (corner c = x | y<<1 | z<<2) whose matter and whose void are each face-connected
    /// inside the block. When every block is like that, the dual surface (one vertex per block) is a 2-manifold.
    static let goodBlock: [Bool] = (0..<256).map { connected($0) && connected(~$0 & 255) }

    static func connected(_ m: Int) -> Bool {
        guard m != 0 else { return true }
        var seen = m & -m, changed = true
        while changed {
            changed = false
            for c in 0..<8 where seen & (1 << c) != 0 {
                for b in 0..<3 {
                    let n = c ^ (1 << b)
                    if m & (1 << n) != 0 && seen & (1 << n) == 0 { seen |= 1 << n; changed = true }
                }
            }
        }
        return seen == m
    }

    @inline(__always) private func blockMask(_ x: Int, _ y: Int, _ z: Int) -> Int {
        var m = 0
        for c in 0..<8 where v[index(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1))] != 0 { m |= 1 << c }
        return m
    }

    /// Adds matter until no 2×2×2 block is ambiguous (two pieces of matter or of void touching only along an
    /// edge or at a corner). Returns the voxels added. Adding, never removing: thin joints get thicker.
    @discardableResult
    mutating func makeWellComposed() -> Int {
        var added = 0
        var work: [SIMD3<Int32>] = []
        func fix(_ x: Int, _ y: Int, _ z: Int) {
            let m = blockMask(x, y, z)
            guard !Self.goodBlock[m] else { return }
            var fill: [Int] = []
            if let c = (0..<8).first(where: { m & (1 << $0) == 0 && Self.goodBlock[m | (1 << $0)] }) { fill = [c] }
            else { fill = (0..<8).filter { m & (1 << $0) == 0 } }
            for c in fill {
                let (vx, vy, vz) = (x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1))
                guard vx > 0, vy > 0, vz > 0, vx < nx - 1, vy < ny - 1, vz < nz - 1 else { continue }
                v[index(vx, vy, vz)] = 1; added += 1
                for dz in -1...0 { for dy in -1...0 { for dx in -1...0 {
                    let (bx, by, bz) = (vx + dx, vy + dy, vz + dz)
                    if bx >= 0, by >= 0, bz >= 0, bx < nx - 1, by < ny - 1, bz < nz - 1 {
                        work.append(SIMD3(Int32(bx), Int32(by), Int32(bz)))
                    }
                } } }
            }
        }
        for z in 0..<(nz - 1) { for y in 0..<(ny - 1) { for x in 0..<(nx - 1) { fix(x, y, z) } } }
        while let b = work.popLast() { fix(Int(b.x), Int(b.y), Int(b.z)) }
        return added
    }
}
