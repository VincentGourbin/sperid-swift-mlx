// Resample.swift - `F.interpolate(mode='bilinear', align_corners=True)` and torchvision's bilinear sampling
// Copyright 2026 Vincent Gourbin

import MLX

enum Resample {
    /// Bilinear resize of NHWC `x` to (h, w), `align_corners=True`: source = dst · (in − 1)/(out − 1)
    /// (0 when out = 1). Separable: rows then columns, weights in fp32.
    static func bilinear(_ x: MLXArray, height h: Int, width w: Int) -> MLXArray {
        var y = axis(x, axis: 1, out: h)
        y = axis(y, axis: 2, out: w)
        return y
    }

    private static func axis(_ x: MLXArray, axis: Int, out: Int) -> MLXArray {
        let n = x.dim(axis)
        if n == out { return x }
        var lo = [Int32](), hi = [Int32](), frac = [Float]()
        for i in 0..<out {
            let s = out == 1 ? 0 : Double(i) * Double(n - 1) / Double(out - 1)
            let l = min(Int(s.rounded(.down)), n - 1)
            lo.append(Int32(l)); hi.append(Int32(min(l + 1, n - 1))); frac.append(Float(s - Double(l)))
        }
        let a = x.take(MLXArray(lo), axis: axis), b = x.take(MLXArray(hi), axis: axis)
        var shape = [Int](repeating: 1, count: x.ndim); shape[axis] = out
        let t = MLXArray(frac).reshaped(shape)
        return (a.asType(.float32) * (1 - t) + b.asType(.float32) * t).asType(x.dtype)
    }
}

/// `torchvision.ops.deform_conv2d` with a modulation mask (DCNv2), stride 1, dilation 1, one offset group.
/// x NHWC [1, H, W, C]; offset NHWC [1, H, W, 2·k²] as (dy, dx) per kernel position (row-major);
/// mask [1, H, W, k²]; weight MLX conv layout [O, k, k, C]. Samples outside the image read 0.
enum DeformConv {
    static func apply(_ x: MLXArray, offset: MLXArray, mask: MLXArray, weight: MLXArray, bias: MLXArray?,
                      kernel k: Int, padding p: Int) -> MLXArray {
        let (h, w, c) = (x.dim(1), x.dim(2), x.dim(3))
        let flat = x.reshaped(h * w, c).asType(.float32)
        let ys = MLXArray((0..<h).map { Float($0) }).reshaped(h, 1)
        let xs = MLXArray((0..<w).map { Float($0) }).reshaped(1, w)
        var samples: [MLXArray] = []
        for ky in 0..<k {
            for kx in 0..<k {
                let i = ky * k + kx
                let py = ys + Float(ky - p) + offset[0, 0..., 0..., 2 * i].asType(.float32)       // [H, W]
                let px = xs + Float(kx - p) + offset[0, 0..., 0..., 2 * i + 1].asType(.float32)
                let v = sampleBilinear(flat, height: h, width: w, y: py, x: px)                 // [H·W, C]
                samples.append(v * mask[0, 0..., 0..., i].asType(.float32).reshaped(h * w, 1))
            }
        }
        // [H·W, k·k·C] × [k·k·C, O]  (weight [O, ky, kx, C] flattened in the same order)
        let cols = concatenated(samples, axis: 1)
        let wmat = weight.asType(.float32).reshaped(weight.dim(0), -1).transposed()
        var y = matmul(cols, wmat)
        if let bias { y = y + bias.asType(.float32) }
        return y.reshaped(1, h, w, weight.dim(0)).asType(x.dtype)
    }

    /// torchvision `bilinear_interpolate`: 0 when y ≤ −1, y ≥ H, x ≤ −1 or x ≥ W; corners outside read 0.
    static func sampleBilinear(_ flat: MLXArray, height h: Int, width w: Int, y: MLXArray, x: MLXArray) -> MLXArray {
        let valid = (y .> -1) .&& (y .< Float(h)) .&& (x .> -1) .&& (x .< Float(w))
        let y0 = floor(y), x0 = floor(x)
        let ly = y - y0, lx = x - x0
        var acc = MLXArray.zeros([h * w, flat.dim(1)], dtype: .float32)
        for (dy, dx, wt) in [(0, 0, (1 - ly) * (1 - lx)), (0, 1, (1 - ly) * lx), (1, 0, ly * (1 - lx)), (1, 1, ly * lx)] {
            let yy = y0 + Float(dy), xx = x0 + Float(dx)
            let inside = valid .&& (yy .>= 0) .&& (yy .<= Float(h - 1)) .&& (xx .>= 0) .&& (xx .<= Float(w - 1))
            let idx = (clip(yy, min: 0, max: Float(h - 1)) * Float(w) + clip(xx, min: 0, max: Float(w - 1))).asType(.int32)
            let gathered = flat.take(idx.reshaped(-1), axis: 0)                                  // [H·W, C]
            acc = acc + gathered * which(inside, wt, MLXArray(Float(0))).reshaped(-1, 1)
        }
        return acc
    }
}
