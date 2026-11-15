// ImagePreprocessor.swift - the image the encoders see (`preprocess.py`, `preprocess_image`)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import SperidCore

public enum ImagePreprocessor {
    /// 1. short side → `size` (Lanczos, Python `round`); 2. alpha from the image or `remover`;
    /// 3. alpha eroded by 1 px (3×3 min); 4. square of side max(w, h)·margin around the alpha's bounding
    /// box (outside = transparent); 5. → `size`² (Lanczos); 6. composited on black.
    public static func prepare(_ image: RGBAImage, margin: Double, size: Int = 1024,
                               remover: BackgroundRemover?) throws -> RGBAImage {
        let scale = Double(size) / Double(min(image.width, image.height))
        let w = max(1, Int((Double(image.width) * scale).rounded(.toNearestOrEven)))
        let h = max(1, Int((Double(image.height) * scale).rounded(.toNearestOrEven)))
        var img = image.resized(width: w, height: h)
        if !img.hasRealAlpha {
            guard let remover else { throw SperidError.invalidRequest("opaque image and no background remover") }
            let alpha = try remover.alpha(for: img)
            for i in 0..<(w * h) { img.pixels[4 * i + 3] = alpha[i] }
        }
        // erode the alpha (PIL MinFilter(3): 3×3 window, edges replicate)
        let alpha = (0..<(w * h)).map { img.pixels[4 * $0 + 3] }
        var left = w, top = h, right = -1, bottom = -1
        for y in 0..<h {
            for x in 0..<w {
                var m: UInt8 = 255
                for dy in -1...1 {
                    let yy = min(h - 1, max(0, y + dy))
                    for dx in -1...1 { m = min(m, alpha[yy * w + min(w - 1, max(0, x + dx))]) }
                }
                img.pixels[4 * (y * w + x) + 3] = m
                if m > 0 { left = min(left, x); right = max(right, x); top = min(top, y); bottom = max(bottom, y) }
            }
        }
        guard right >= 0 else { throw SperidError.invalidRequest("no object found in the image (the alpha channel is empty)") }
        let cx = Double(left + right) / 2, cy = Double(top + bottom) / 2
        let half = Double(max(right - left, bottom - top)) / 2 * margin
        let x0 = Int(cx - half), y0 = Int(cy - half), x1 = Int(cx + half), y1 = Int(cy + half)
        var crop = RGBAImage(width: x1 - x0, height: y1 - y0)
        for y in 0..<crop.height {
            let sy = y + y0
            guard sy >= 0 && sy < h else { continue }
            for x in 0..<crop.width {
                let sx = x + x0
                guard sx >= 0 && sx < w else { continue }
                for c in 0..<4 { crop.pixels[4 * (y * crop.width + x) + c] = img.pixels[4 * (sy * w + sx) + c] }
            }
        }
        var out = crop.resized(width: size, height: size)
        // composite on black: rgb · a (PIL paste with the alpha as mask)
        for i in 0..<(size * size) {
            let a = UInt16(out.pixels[4 * i + 3])
            for c in 0..<3 { out.pixels[4 * i + c] = UInt8((UInt16(out.pixels[4 * i + c]) * a + 127) / 255) }
            out.pixels[4 * i + 3] = 255
        }
        return out
    }

    /// [1, 3, H, W] ∈ [0, 1] float32, as `torch.from_numpy(np.array(img)).permute(2, 0, 1) / 255`.
    public static func pixels(_ image: RGBAImage) -> MLXArray {
        let rgba = MLXArray(image.pixels, [image.height, image.width, 4]).asType(.float32) / 255
        return rgba[0..., 0..., 0..<3].transposed(2, 0, 1).expandedDimensions(axis: 0)
    }
}
