// RGBAImage.swift - a minimal 8-bit RGBA raster with CoreGraphics / vImage conversions
// Copyright 2026 Vincent Gourbin

import Accelerate
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Straight (non-premultiplied) 8-bit RGBA, row-major. The preprocessing works on this, like PIL on "RGBA".
public struct RGBAImage: Sendable {
    public var width: Int
    public var height: Int
    public var pixels: [UInt8]  // width · height · 4

    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height * 4)
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    public init(width: Int, height: Int) { self.init(width: width, height: height, pixels: [UInt8](repeating: 0, count: width * height * 4)) }

    /// Decodes any image ImageIO reads (PNG, JPEG, WebP, HEIC…), keeping its alpha.
    public init(contentsOf url: URL) throws {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw CocoaError(.fileReadCorruptFile) }
        self.init(cgImage: image)
    }

    public init(cgImage: CGImage) {
        let w = cgImage.width, h = cgImage.height
        width = w
        height = h
        var buffer = [UInt8](repeating: 0, count: w * h * 4)
        // draw premultiplied (CoreGraphics' only RGBA option), then un-premultiply
        buffer.withUnsafeMutableBytes { raw in
            let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        for i in stride(from: 0, to: buffer.count, by: 4) where buffer[i + 3] != 0 && buffer[i + 3] != 255 {
            let a = Float(buffer[i + 3])
            for c in 0..<3 { buffer[i + c] = UInt8(min(255, (Float(buffer[i + c]) * 255 / a).rounded())) }
        }
        pixels = buffer
    }

    /// `has_real_alpha`: an alpha channel with at least one pixel under 255.
    public var hasRealAlpha: Bool { stride(from: 3, to: pixels.count, by: 4).contains { pixels[$0] < 255 } }

    public func cgImage(premultipliedOnBlack: Bool = false) -> CGImage {
        let data = CFDataCreate(nil, pixels, pixels.count)!
        let info: CGImageAlphaInfo = premultipliedOnBlack ? .noneSkipLast : .last
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: info.rawValue),
                       provider: CGDataProvider(data: data)!, decode: nil, shouldInterpolate: true, intent: .defaultIntent)!
    }

    public func writePNG(to url: URL, opaque: Bool = false) throws {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(dest, cgImage(premultipliedOnBlack: opaque), nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }

    /// High-quality (Lanczos) resampling through vImage, on premultiplied pixels like PIL does for RGBA.
    public func resized(width w: Int, height h: Int) -> RGBAImage {
        var src = premultiplied()
        var out = [UInt8](repeating: 0, count: w * h * 4)
        src.withUnsafeMutableBytes { s in
            out.withUnsafeMutableBytes { d in
                var sb = vImage_Buffer(data: s.baseAddress, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width * 4)
                var db = vImage_Buffer(data: d.baseAddress, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)
                _ = vImageScale_ARGB8888(&sb, &db, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }
        return RGBAImage(width: w, height: h, pixels: out).unpremultiplied()
    }

    func premultiplied() -> [UInt8] {
        var p = pixels
        for i in stride(from: 0, to: p.count, by: 4) {
            let a = UInt16(p[i + 3])
            for c in 0..<3 { p[i + c] = UInt8((UInt16(p[i + c]) * a + 127) / 255) }
        }
        return p
    }

    func unpremultiplied() -> RGBAImage {
        var p = pixels
        for i in stride(from: 0, to: p.count, by: 4) {
            let a = Float(p[i + 3])
            for c in 0..<3 { p[i + c] = a == 0 ? 0 : UInt8(min(255, (Float(p[i + c]) * 255 / a).rounded())) }
        }
        return RGBAImage(width: width, height: height, pixels: p)
    }
}
