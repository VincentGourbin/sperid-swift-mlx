// SplatEntity.swift - a GaussianCloud as a RealityKit 27 Gaussian splat entity
// Copyright 2026 Vincent Gourbin

import IdolCore
import Metal
import RealityKit
import simd

/// One interleaved buffer per cloud, the same values a 3DGS `.ply` stores (y-up position, log scale,
/// unit quaternion, logit opacity, SH degree 0), with RealityKit's exponential / sigmoid activations.
@available(macOS 27.0, *)
@MainActor
enum SplatEntity {
    /// RealityKit's quaternion layout is not documented in the interface: simd order (x, y, z, w) by
    /// default; `wFirst` switches to the PLY order (w, x, y, z) if a render shows twisted splats.
    static var wFirst = false

    static let stride = MemoryLayout<Float>.size * (3 + 3 + 4 + 1 + 3)

    static func make(_ cloud: GaussianCloud) throws -> Entity {
        let n = cloud.count
        let buffer = try LowLevelBuffer(descriptor: .init(capacity: n * stride))
        buffer.replaceUnsafeMutableBytes { raw in
            let f = raw.bindMemory(to: Float.self)
            for i in 0..<n {
                let v = PLYWriter.vertex(cloud, i)   // x y z nx ny nz f_dc×3 opacity scale×3 rot×4
                var o = i * 14
                for k in [0, 1, 2] { f[o] = v[k]; o += 1 }                // position
                for k in [10, 11, 12] { f[o] = v[k]; o += 1 }             // log scale
                let (w, x, y, z) = (v[13], v[14], v[15], v[16])
                for q in (wFirst ? [w, x, y, z] : [x, y, z, w]) { f[o] = q; o += 1 }
                f[o] = v[9]; o += 1                                        // logit opacity
                for k in [6, 7, 8] { f[o] = v[k]; o += 1 }                 // SH DC
            }
        }
        func descriptor(_ format: MTLAttributeFormat, _ floatOffset: Int) -> GaussianSplatResource.BufferDescriptor {
            .init(buffer: buffer, format: format, stride: stride, offset: floatOffset * MemoryLayout<Float>.size)
        }
        let resource = try GaussianSplatResource.BufferResource(
            count: n, position: descriptor(.float3, 0), scale: descriptor(.float3, 3),
            rotation: descriptor(.float4, 6), opacity: descriptor(.float, 10),
            sphericalHarmonics: (descriptor(.float3, 11), .zero))
        let splat = GaussianSplatResource(resource)
        splat.scaleActivation = .exponential
        splat.opacityActivation = .sigmoid
        let entity = Entity()
        entity.components.set(GaussianSplatComponent(splat))
        // the 3DGS PLY convention (`save_ply`: (x, −z, y)) is y-DOWN (COLMAP); RealityKit is y-up:
        // a half turn about x puts the object upright, (x, −z, y) → (x, z, −y)
        entity.orientation = simd_quatf(angle: .pi, axis: [1, 0, 0])
        return entity
    }
}
