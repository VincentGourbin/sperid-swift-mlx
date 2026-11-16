// TriangleMesh.swift - closed triangle meshes for printing: extraction from a solid, smoothing, decimation, checks
// Copyright 2026 Vincent Gourbin

import Foundation
import simd

/// An indexed triangle mesh (counter-clockwise seen from outside).
struct TriangleMesh {
    var positions: [SIMD3<Double>] = []
    var triangles: [SIMD3<Int32>] = []

    // MARK: - Extraction

    /// Surface nets on a well-composed solid: the topology comes from the voxels (closed 2-manifold), the vertex
    /// positions from `smooth` (the solid blurred, crossing 0.5); Taubin smoothing then removes the voxel steps.
    /// Positions in grid cells; quads counter-clockwise seen from outside.
    static func surfaceQuads(_ solid: VoxelSolid, smooth: [Float])
        -> (positions: [SIMD3<Double>], quads: [SIMD4<Int32>]) {
        let (nx, ny, nz) = (solid.nx, solid.ny, solid.nz)
        func inside(_ x: Int, _ y: Int, _ z: Int) -> Bool { solid.v[solid.index(x, y, z)] != 0 }
        var vertexOf = [Int32](repeating: -1, count: (nx - 1) * (ny - 1) * (nz - 1))
        func cell(_ x: Int, _ y: Int, _ z: Int) -> Int { (z * (ny - 1) + y) * (nx - 1) + x }
        var positions: [SIMD3<Double>] = []
        var quads: [SIMD4<Int32>] = []
        for z in 0..<(nz - 1) { for y in 0..<(ny - 1) { for x in 0..<(nx - 1) {
            var mask = 0
            for c in 0..<8 where inside(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1)) { mask |= 1 << c }
            guard mask != 0 && mask != 255 else { continue }
            var val = [Float](repeating: 0, count: 8)
            for c in 0..<8 { val[c] = smooth[solid.index(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1))] }
            var sum = SIMD3<Double>.zero, crossings = 0.0
            for (a, b) in SurfaceNets.cubeEdges where (mask >> a) & 1 != (mask >> b) & 1 {
                var t: Float = 0.5
                if (val[a] - 0.5) * (val[b] - 0.5) < 0 { t = min(0.9, max(0.1, (0.5 - val[a]) / (val[b] - val[a]))) }
                let ca = SIMD3<Double>(SurfaceNets.corner(a)), cb = SIMD3<Double>(SurfaceNets.corner(b))
                sum += ca + (cb - ca) * Double(t)
                crossings += 1
            }
            vertexOf[cell(x, y, z)] = Int32(positions.count)
            positions.append(SIMD3(Double(x), Double(y), Double(z)) + sum / crossings)
        } } }
        for z in 1..<(nz - 1) { for y in 1..<(ny - 1) { for x in 1..<(nx - 1) {
            let isIn = inside(x, y, z)
            for axis in 0..<3 {
                let (ex, ey, ez) = axis == 0 ? (1, 0, 0) : (axis == 1 ? (0, 1, 0) : (0, 0, 1))
                guard isIn != inside(x + ex, y + ey, z + ez) else { continue }
                let cells: [(Int, Int, Int)]
                switch axis {
                case 0: cells = [(x, y - 1, z - 1), (x, y, z - 1), (x, y, z), (x, y - 1, z)]
                case 1: cells = [(x - 1, y, z - 1), (x - 1, y, z), (x, y, z), (x, y, z - 1)]
                default: cells = [(x - 1, y - 1, z), (x, y - 1, z), (x, y, z), (x - 1, y, z)]
                }
                var q = cells.map { vertexOf[cell($0.0, $0.1, $0.2)] }
                guard q.allSatisfy({ $0 >= 0 }) else { continue }
                if !isIn { q.reverse() }
                quads.append(SIMD4(q[0], q[1], q[2], q[3]))
            }
        } } }
        return (positions, quads)
    }

    static func extract(_ solid: VoxelSolid, smooth: [Float]) -> TriangleMesh {
        let (positions, quads) = surfaceQuads(solid, smooth: smooth)
        var mesh = TriangleMesh(positions: positions)
        mesh.triangles.reserveCapacity(2 * quads.count)
        for q in quads {
            // split along the shorter diagonal
            let d02 = simd_length_squared(positions[Int(q[0])] - positions[Int(q[2])])
            let d13 = simd_length_squared(positions[Int(q[1])] - positions[Int(q[3])])
            if d02 <= d13 {
                mesh.triangles.append(SIMD3(q[0], q[1], q[2])); mesh.triangles.append(SIMD3(q[0], q[2], q[3]))
            } else {
                mesh.triangles.append(SIMD3(q[0], q[1], q[3])); mesh.triangles.append(SIMD3(q[1], q[2], q[3]))
            }
        }
        return mesh
    }

    // MARK: - Smoothing

    /// Taubin λ|μ smoothing (uniform weights): removes the voxel ripple without shrinking the object.
    mutating func taubin(iterations: Int, lambda: Double = 0.5, mu: Double = -0.53) {
        Self.taubin(&positions, edges: triangles.flatMap { [($0.x, $0.y), ($0.y, $0.z), ($0.z, $0.x)] },
                    iterations: iterations, lambda: lambda, mu: mu)
    }

    static func taubin(_ positions: inout [SIMD3<Double>], edges: [(Int32, Int32)], iterations: Int,
                       lambda: Double = 0.5, mu: Double = -0.53) {
        guard iterations > 0 else { return }
        var neighbours = [[Int32]](repeating: [], count: positions.count)
        for (a, b) in edges {
            if !neighbours[Int(a)].contains(b) { neighbours[Int(a)].append(b) }
            if !neighbours[Int(b)].contains(a) { neighbours[Int(b)].append(a) }
        }
        var next = positions
        for _ in 0..<iterations {
            for factor in [lambda, mu] {
                positions.withUnsafeBufferPointer { cur in
                    next.withUnsafeMutableBufferPointer { out in
                        DispatchQueue.concurrentPerform(iterations: 64) { chunk in
                            let n = cur.count, a = chunk * n / 64, e = (chunk + 1) * n / 64
                            for i in a..<e where !neighbours[i].isEmpty {
                                var mean = SIMD3<Double>.zero
                                for j in neighbours[i] { mean += cur[Int(j)] }
                                mean /= Double(neighbours[i].count)
                                out[i] = cur[i] + factor * (mean - cur[i])
                            }
                        }
                    }
                }
                swap(&positions, &next)
            }
        }
    }

    // MARK: - Checks

    struct Check {
        var watertight: Bool     // every edge has exactly two triangles, in opposite directions
        var manifold: Bool       // and the triangles around every vertex form a single fan
        var shells: Int
        var genus: Int
        var volume: Double       // signed: > 0 when the triangles face outwards
    }

    func check() -> Check {
        var directed = Set<UInt64>(minimumCapacity: triangles.count * 3)
        var repeated = false
        for t in triangles {
            for (a, b) in [(t.x, t.y), (t.y, t.z), (t.z, t.x)] {
                if !directed.insert(UInt64(UInt32(a)) << 32 | UInt64(UInt32(b))).inserted { repeated = true }
            }
        }
        var watertight = !repeated
        if watertight {
            for e in directed where !directed.contains((e << 32) | (e >> 32)) { watertight = false; break }
        }
        // fans: around v, the edges opposite to v chain into one cycle
        var faces = [[Int32]](repeating: [], count: positions.count)
        for (i, t) in triangles.enumerated() { for v in [t.x, t.y, t.z] { faces[Int(v)].append(Int32(i)) } }
        var manifold = watertight
        if manifold {
            for v in positions.indices where !faces[v].isEmpty {
                var next: [Int32: Int32] = [:]
                for f in faces[v] {
                    let t = triangles[Int(f)]
                    let (b, c) = t.x == Int32(v) ? (t.y, t.z) : (t.y == Int32(v) ? (t.z, t.x) : (t.x, t.y))
                    next[b] = c
                }
                guard let start = next.keys.first else { continue }
                var cur = start, steps = 0
                repeat { guard let n = next[cur] else { break }; cur = n; steps += 1 } while cur != start && steps <= next.count
                if steps != faces[v].count || cur != start { manifold = false; break }
            }
        }
        // shells (union-find over vertices)
        var parent = Array(0..<Int32(positions.count))
        func find(_ x: Int32) -> Int32 {
            var x = x
            while parent[Int(x)] != x { parent[Int(x)] = parent[Int(parent[Int(x)])]; x = parent[Int(x)] }
            return x
        }
        for t in triangles {
            let a = find(t.x)
            for b in [find(t.y), find(t.z)] where b != a { parent[Int(b)] = a }
        }
        var roots = Set<Int32>()
        var used = 0
        for v in positions.indices where !faces[v].isEmpty { roots.insert(find(Int32(v))); used += 1 }
        let euler = used - directed.count / 2 + triangles.count
        var volume = 0.0
        for t in triangles {
            volume += simd_dot(positions[Int(t.x)], simd_cross(positions[Int(t.y)], positions[Int(t.z)])) / 6
        }
        return Check(watertight: watertight, manifold: manifold, shells: roots.count,
                     genus: (2 * roots.count - euler) / 2, volume: volume)
    }
}

// MARK: - Decimation

/// Garland-Heckbert quadric edge collapse for closed manifolds. A collapse is refused when it would break the
/// manifold (link condition), flip or squash a triangle, so the result stays printable.
enum Decimation {
    /// Upper triangle of the symmetric 4×4 error matrix, stored inline (no allocation in the hot loop).
    struct Quadric {
        var m = SIMD8<Double>()      // xx xy xz xw yy yz yw zz
        var r = SIMD2<Double>()      // zw ww
        static func plane(_ n: SIMD3<Double>, _ d: Double, weight w: Double) -> Quadric {
            Quadric(m: SIMD8(n.x * n.x, n.x * n.y, n.x * n.z, n.x * d, n.y * n.y, n.y * n.z, n.y * d, n.z * n.z) * w,
                    r: SIMD2(n.z * d, d * d) * w)
        }
        static func + (l: Quadric, r: Quadric) -> Quadric { Quadric(m: l.m + r.m, r: l.r + r.r) }
        func error(_ p: SIMD3<Double>) -> Double {
            let (x, y, z) = (p.x, p.y, p.z)
            return m[0] * x * x + 2 * m[1] * x * y + 2 * m[2] * x * z + 2 * m[3] * x
                + m[4] * y * y + 2 * m[5] * y * z + 2 * m[6] * y + m[7] * z * z + 2 * r[0] * z + r[1]
        }
        /// The point minimising the error, if the 3×3 system is well conditioned.
        func optimum() -> SIMD3<Double>? {
            let a = simd_double3x3(rows: [SIMD3(m[0], m[1], m[2]), SIMD3(m[1], m[4], m[5]), SIMD3(m[2], m[5], m[7])])
            let scale = m[0] + m[4] + m[7]
            guard scale > 0, abs(a.determinant) > 1e-10 * scale * scale * scale else { return nil }
            return a.inverse * SIMD3(-m[3], -m[6], -r[0])
        }
    }

    struct Candidate: Comparable {
        var cost: Double, a: Int32, b: Int32, va: UInt32, vb: UInt32, p: SIMD3<Double>
        static func < (l: Candidate, r: Candidate) -> Bool { l.cost < r.cost }
        static func == (l: Candidate, r: Candidate) -> Bool { l.cost == r.cost }
    }

    struct MinHeap {
        var items: [Candidate] = []
        mutating func push(_ c: Candidate) {
            items.append(c)
            var i = items.count - 1
            while i > 0 { let p = (i - 1) / 2; if items[i] < items[p] { items.swapAt(i, p); i = p } else { break } }
        }
        mutating func pop() -> Candidate? {
            guard let first = items.first else { return nil }
            let last = items.removeLast()
            if !items.isEmpty {
                items[0] = last
                var i = 0
                while true {
                    let l = 2 * i + 1, r = l + 1
                    var m = i
                    if l < items.count && items[l] < items[m] { m = l }
                    if r < items.count && items[r] < items[m] { m = r }
                    if m == i { break }
                    items.swapAt(i, m); i = m
                }
            }
            return first
        }
    }

    static func run(_ mesh: inout TriangleMesh, targetTriangles: Int) {
        guard mesh.triangles.count > targetTriangles else { return }
        var pos = mesh.positions
        var tris = mesh.triangles
        var triAlive = [Bool](repeating: true, count: tris.count)
        var vtris = [[Int32]](repeating: [], count: pos.count)
        for (i, t) in tris.enumerated() { for v in [t.x, t.y, t.z] { vtris[Int(v)].append(Int32(i)) } }
        var version = [UInt32](repeating: 0, count: pos.count)
        var alive = [Bool](repeating: true, count: pos.count)
        func normal(_ t: SIMD3<Int32>, _ p: [SIMD3<Double>]) -> SIMD3<Double> {
            simd_cross(p[Int(t.y)] - p[Int(t.x)], p[Int(t.z)] - p[Int(t.x)])
        }
        var quadric = [Quadric](repeating: Quadric(), count: pos.count)
        for t in tris {
            let n = normal(t, pos)
            let area = simd_length(n)
            guard area > 1e-12 else { continue }
            let u = n / area
            let q = Quadric.plane(u, -simd_dot(u, pos[Int(t.x)]), weight: area / 2)
            for v in [t.x, t.y, t.z] { quadric[Int(v)] = quadric[Int(v)] + q }
        }
        func neighbours(_ v: Int32) -> [Int32] {
            var out: [Int32] = []
            for f in vtris[Int(v)] { for w in [tris[Int(f)].x, tris[Int(f)].y, tris[Int(f)].z] where w != v && !out.contains(w) { out.append(w) } }
            return out
        }
        func candidate(_ a: Int32, _ b: Int32) -> Candidate {
            let q = quadric[Int(a)] + quadric[Int(b)]
            let pa = pos[Int(a)], pb = pos[Int(b)], mid = (pa + pb) / 2
            var best = mid, cost = q.error(mid)
            for p in [pa, pb] { let c = q.error(p); if c < cost { cost = c; best = p } }
            if let o = q.optimum(), simd_length(o - mid) <= simd_length(pb - pa) {
                let c = q.error(o); if c < cost { cost = c; best = o }
            }
            return Candidate(cost: max(0, cost), a: a, b: b, va: version[Int(a)], vb: version[Int(b)], p: best)
        }
        var heap = MinHeap()
        heap.items.reserveCapacity(tris.count * 2)
        for t in tris { for (a, b) in [(t.x, t.y), (t.y, t.z), (t.z, t.x)] where a < b { heap.push(candidate(a, b)) } }
        var count = tris.count
        while count > targetTriangles, let c = heap.pop() {
            let (a, b) = (c.a, c.b)
            guard alive[Int(a)], alive[Int(b)], version[Int(a)] == c.va, version[Int(b)] == c.vb else { continue }
            // link condition: on a closed manifold, a and b share exactly the two vertices opposite their edge
            let na = neighbours(a), nb = neighbours(b)
            guard na.contains(b), na.count > 3, nb.count > 3 else { continue }
            let common = na.filter { nb.contains($0) }
            guard common.count == 2 else { continue }
            // no flipped or squashed triangle around a or b
            var ok = true
            for v in [a, b] where ok {
                for f in vtris[Int(v)] {
                    let t = tris[Int(f)]
                    let hasA = t.x == a || t.y == a || t.z == a, hasB = t.x == b || t.y == b || t.z == b
                    if hasA && hasB { continue }
                    let n0 = normal(t, pos)
                    let corner = { (w: Int32) -> SIMD3<Double> in (w == v) ? c.p : pos[Int(w)] }
                    let n1 = simd_cross(corner(t.y) - corner(t.x), corner(t.z) - corner(t.x))
                    let l0 = simd_length(n0), l1 = simd_length(n1)
                    if l1 < 1e-9 || simd_dot(n0, n1) < 0.3 * l0 * l1 { ok = false; break }
                }
            }
            guard ok else { continue }
            // collapse b into a
            for f in vtris[Int(b)] {
                var t = tris[Int(f)]
                if t.x == a || t.y == a || t.z == a {
                    triAlive[Int(f)] = false; count -= 1
                    for w in [t.x, t.y, t.z] where w != b { vtris[Int(w)].removeAll { $0 == f } }
                } else {
                    if t.x == b { t.x = a } else if t.y == b { t.y = a } else { t.z = a }
                    tris[Int(f)] = t
                    vtris[Int(a)].append(f)
                }
            }
            vtris[Int(b)] = []
            alive[Int(b)] = false
            pos[Int(a)] = c.p
            quadric[Int(a)] = quadric[Int(a)] + quadric[Int(b)]
            version[Int(a)] &+= 1
            // only a's quadric and position changed: re-cost its edges (the others are re-checked when popped)
            for n in neighbours(a) { heap.push(candidate(min(a, n), max(a, n))) }
        }
        // compact
        var remap = [Int32](repeating: -1, count: pos.count)
        var out = TriangleMesh()
        for (i, t) in tris.enumerated() where triAlive[i] {
            var r = SIMD3<Int32>()
            for k in 0..<3 {
                let v = Int(t[k])
                if remap[v] < 0 { remap[v] = Int32(out.positions.count); out.positions.append(pos[v]) }
                r[k] = remap[v]
            }
            out.triangles.append(r)
        }
        mesh = out
    }
}
