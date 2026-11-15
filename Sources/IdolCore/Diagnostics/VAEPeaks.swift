// VAEPeaks.swift - `idol profile vae-peaks`: MLX active-memory peak of each stage of the VAE encoder
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

public enum VAEPeaks {
    /// Runs the encoder on a 1024² input stage by stage, resetting the peak before each; returns
    /// (stage, peak GB, ms).
    public static func run(_ vae: FluxVAEEncoder, size: Int = 1024) -> [(String, Double, Double)] {
        var rows: [(String, Double, Double)] = []
        func stage(_ name: String, _ body: () -> MLXArray) -> MLXArray {
            Memory.clearCache()
            Memory.peakMemory = 0
            let t0 = DispatchTime.now().uptimeNanoseconds
            let out = body(); eval(out)
            rows.append((name, Double(Memory.peakMemory) / 1e9, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6))
            return out
        }
        let enc = vae.encoder
        let x = MLXRandom.uniform(low: -1, high: 1, [1, size, size, 3]).asType(vae.computeDType); eval(x)
        var h = stage("conv_in") { enc.convIn(x) }
        for (i, block) in enc.downBlocks.enumerated() {
            for (j, r) in block.resnets.enumerated() { h = stage("down \(i) resnet \(j) @\(h.dim(1))²") { r(h) } }
            for d in block.downsamplers { h = stage("down \(i) downsample") { d(h) } }
        }
        h = stage("mid resnet 0") { enc.midBlock.resnets[0](h) }
        h = stage("mid attention (\(h.dim(1) * h.dim(2)) tokens × \(h.dim(3)))") { enc.midBlock.attentions[0](h) }
        h = stage("mid resnet 1") { enc.midBlock.resnets[1](h) }
        _ = stage("norm_out + conv_out") { enc.convOut(silu(enc.normOut(h))) }
        return rows
    }
}
