// BackgroundRemover.swift - who predicts the alpha of an image without one
// Copyright 2026 Vincent Gourbin

import CoreGraphics
import CoreVideo
import Foundation
import Vision

/// Predicts an alpha mask (0…255, image size) for an opaque image. The reference uses BiRefNet
/// (`models/birefnet.py`); an image that already has real alpha never reaches a remover.
public protocol BackgroundRemover: Sendable {
    func alpha(for image: RGBAImage) throws -> [UInt8]
}

/// Apple Vision's subject lifting (`VNGenerateForegroundInstanceMaskRequest`, all instances): fully native,
/// no weights. Not BiRefNet: masks differ at the edges (plan/14 Q-3); the default until BiRefNet is ported.
public struct VisionBackgroundRemover: BackgroundRemover {
    public init() {}

    public func alpha(for image: RGBAImage) throws -> [UInt8] {
        let handler = VNImageRequestHandler(cgImage: image.cgImage(), options: [:])
        let request = VNGenerateForegroundInstanceMaskRequest()
        try handler.perform([request])
        guard let observation = request.results?.first else { return [UInt8](repeating: 0, count: image.width * image.height) }
        let buffer = try observation.generateScaledMaskForImage(forInstances: observation.allInstances, from: handler)
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        precondition(w == image.width && h == image.height, "Vision mask size differs from the image")
        var alpha = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
            let row = UnsafeRawPointer(base + y * rowBytes).assumingMemoryBound(to: Float32.self)
            for x in 0..<w { alpha[y * w + x] = UInt8(max(0, min(255, (row[x] * 255).rounded()))) }
        }
        return alpha
    }
}
