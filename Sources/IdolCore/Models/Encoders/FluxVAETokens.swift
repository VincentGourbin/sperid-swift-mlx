// FluxVAETokens.swift - FLUX.2 VAE encoder → Idol's `feature2` tokens (`conditioning.py`, `_flux2_tokens`)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import SperidCore

/// The FLUX.2 VAE as Idol uses it: encode `pixels·2 − 1`, **sample** the posterior (pitfall 10),
/// patchify 2×2 (channel c·4 + dy·2 + dx), normalise with the checkpoint's BatchNorm statistics
/// (eps 1e-4), flatten to [1, 4096, 128] and prepend 5 zero rows → [1, 4101, 128].
public final class FluxVAEEncoder: Module {
    @ModuleInfo(key: "encoder") var encoder: VAEEncoderNet
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "bn") var bn: VAELatentStats
    public let computeDType: DType
    static let batchNormEps: Float = 1e-4
    public static let latentChannels = 32

    public init(computeDType: DType = .bfloat16) {
        self.computeDType = computeDType
        _encoder.wrappedValue = VAEEncoderNet()
        _quantConv.wrappedValue = Conv2d(inputChannels: 64, outputChannels: 64, kernelSize: 1)
        _bn.wrappedValue = VAELatentStats(4 * Self.latentChannels)
        super.init()
    }

    /// Loads the diffusers `vae/diffusion_pytorch_model.safetensors` (encoder half, `quant_conv`, `bn`):
    /// conv weights [O, I, kh, kw] → [O, kh, kw, I]; the decoder keys are skipped; `computeDType` for
    /// the network (the reference: bf16), BN statistics kept fp32.
    public static func load(url: URL, computeDType: DType = .bfloat16) throws -> FluxVAEEncoder {
        let model = FluxVAEEncoder(computeDType: computeDType)
        var selected: [String: MLXArray] = [:]
        for (key, value) in try loadArrays(url: url) {
            guard key.hasPrefix("encoder.") || key.hasPrefix("quant_conv.") || key.hasPrefix("bn.running") else { continue }
            var v = value
            if v.ndim == 4 { v = v.transposed(0, 2, 3, 1) }
            if !key.hasPrefix("bn.") { v = v.asType(computeDType) }
            eval(v)
            selected[key] = v
        }
        try model.update(parameters: ModuleParameters.unflattened(selected), verify: .all)
        return model
    }

    /// Posterior moments of `pixels` [1, 3, H, W] ∈ [0, 1]: (mean, logvar), NHWC [1, H/8, W/8, 32].
    public func moments(_ pixels: MLXArray) -> (mean: MLXArray, logvar: MLXArray) {
        let x = (pixels.transposed(0, 2, 3, 1).asType(computeDType) * 2 - 1)
        let m = quantConv(encoder(x))
        let c = Self.latentChannels
        let logvar = clip(m[.ellipsis, c...].asType(.float32), min: -30, max: 20).asType(m.dtype)
        return (m[.ellipsis, 0..<c], logvar)
    }

    /// `feature2` [1, 5 + (H/16)², 128]. `noise` is the posterior ε, NCHW [1, 32, H/8, W/8] (the
    /// reference's layout, as recorded) or drawn by the caller.
    public func tokens(_ pixels: MLXArray, noise: MLXArray) -> MLXArray {
        let (mean, logvar) = moments(pixels)
        let eps = noise.transposed(0, 2, 3, 1).asType(mean.dtype)
        let z = mean + exp(0.5 * logvar) * eps                       // [1, h, w, 32] NHWC
        let (b, h, w, c) = (z.dim(0), z.dim(1), z.dim(2), z.dim(3))
        // patchify: [b, h/2, dy, w/2, dx, c] → [b, h/2, w/2, c, dy, dx] → channel c·4 + dy·2 + dx
        let p = z.reshaped(b, h / 2, 2, w / 2, 2, c).transposed(0, 1, 3, 5, 2, 4).reshaped(b, (h / 2) * (w / 2), 4 * c)
        let mean128 = bn.runningMean.asType(p.dtype)
        let std128 = sqrt(bn.runningVar + Self.batchNormEps).asType(p.dtype)
        let tokens = ((p - mean128) / std128).asType(.float32)
        return concatenated([MLXArray.zeros([b, 5, 4 * c]), tokens], axis: 1)
    }
}
