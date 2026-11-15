// OffscreenRender.swift - render a splat to PNG without a window (RealityRenderer → MTLTexture)
// Copyright 2026 Vincent Gourbin

import AppKit
import IdolCore
import Metal
import RealityKit
import simd

/// `idol-studio file.ply --render out.png`: `views` yaw angles around the object, side by side, for
/// checking the viewer from a script (no screen needed) and for thumbnails.
@available(macOS 27.0, *)
@MainActor
enum OffscreenRender {
    /// Eye direction (RealityKit, y-up) of the camera the model predicted for its input image: in the
    /// object's z-up frame d = (sin yaw·cos pitch, cos yaw·cos pitch, sin pitch); y-up: (dx, dz, −dy).
    static func inputDirection(_ pose: CameraPose) -> SIMD3<Float> {
        let (y, p) = (Float(pose.yaw), Float(pose.pitch))
        let d = SIMD3<Float>(sin(y) * cos(p), cos(y) * cos(p), sin(p))
        return SIMD3<Float>(d.x, d.z, -d.y)
    }

    static func render(_ cloud: GaussianCloud, to url: URL, views: Int = 4, size: Int = 512,
                       inputCamera: CameraPose? = nil) async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let renderer = try RealityRenderer()
        renderer.cameraSettings.colorBackground = .color(CGColor(gray: 0.05, alpha: 1))
        renderer.cameraSettings.isToneMappingEnabled = false
        let splat = try SplatEntity.make(cloud)
        renderer.entities.append(splat)
        let camera = PerspectiveCamera()
        camera.camera.fieldOfViewInDegrees = 40
        renderer.entities.append(camera)
        renderer.activeCamera = camera

        // pixel-exact strip (an NSImage would double it on a Retina screen)
        let stripContext = CGContext(data: nil, width: size * views, height: size, bitsPerComponent: 8, bytesPerRow: 0,
                                     space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        var tiles: [CGImage] = []
        for v in 0..<views {
            // view 0: the input camera when known, then turns of 360°/views around the vertical
            let start = inputCamera.map(inputDirection) ?? SIMD3<Float>(0, 0.35, 1)
            let yaw = Float(v) / Float(views) * 2 * .pi
            let turned = SIMD3<Float>(start.x * cos(yaw) + start.z * sin(yaw), start.y, -start.x * sin(yaw) + start.z * cos(yaw))
            let eye = normalize(turned) * 1.9
            camera.look(at: .zero, from: eye, relativeTo: nil)
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm_srgb, width: size, height: size, mipmapped: false)
            desc.usage = [.renderTarget, .shaderRead]
            desc.storageMode = .shared
            let texture = device.makeTexture(descriptor: desc)!
            let output = try RealityRenderer.CameraOutput(.singleProjection(colorTexture: texture))
            // the splat buffers upload and sort on the first frames: render a few, keep the last
            for _ in 0..<3 {
                try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                    do {
                        try renderer.updateAndRender(deltaTime: 1.0 / 60, cameraOutput: output, onComplete: { _ in c.resume() })
                    } catch { c.resume(throwing: error) }
                }
            }
            var bytes = [UInt8](repeating: 0, count: size * size * 4)
            texture.getBytes(&bytes, bytesPerRow: size * 4, from: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0)
            let ctx = CGContext(data: &bytes, width: size, height: size, bitsPerComponent: 8, bytesPerRow: size * 4,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
            tiles.append(ctx.makeImage()!)
        }
        for (i, tile) in tiles.enumerated() {
            stripContext.draw(tile, in: CGRect(x: i * size, y: 0, width: size, height: size))
        }
        let rep = NSBitmapImageRep(cgImage: stripContext.makeImage()!)
        try rep.representation(using: .png, properties: [:])!.write(to: url)
    }
}
