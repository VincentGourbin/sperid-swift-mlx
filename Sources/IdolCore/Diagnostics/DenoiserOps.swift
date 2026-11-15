// DenoiserOps.swift - `idol profile denoiser-ops`: where one joint block's time goes, op by op
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import MLXProfiler

/// Each op of a real joint block (block 12 of the published denoiser, fp16 blocks) is run in isolation on
/// the real shapes (12 294 tokens), evaluated, warm-up excluded; median of `runs`. Sum × 24 ≈ the joint part.
public struct DenoiserOps {
    let model: IdolDenoiser
    let runs: Int

    public init(model: IdolDenoiser, runs: Int) {
        self.model = model
        self.runs = runs
    }

    func median(_ name: String, _ body: () -> MLXArray) -> Double {
        eval(body())                                  // warm-up (kernel compile, allocation)
        var ms: [Double] = []
        for _ in 0..<runs {
            let t0 = DispatchTime.now().uptimeNanoseconds
            eval(body())
            ms.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
        }
        let m = ms.sorted()[ms.count / 2]
        IdolProfiling.complete("op: \(name)", .denoisingStep, microseconds: UInt64(m * 1000))
        return m
    }

    public func run(tokens l: Int) -> [(String, Double)] {
        let block = model.blocks[12], rope = model.rePo[12]
        let c = 1024, h = 16, d = 64, bd = model.blockDType
        let key = MLXRandom.key(1)
        let x = (MLXRandom.normal([1, l, c], key: key) * 0.5).asType(bd)
        let mod = MLXRandom.normal([1, 6 * c], key: key).asType(bd) * 0.1
        let (cosA, sinA) = rope(x); eval(cosA, sinA)
        let m = modulationChunks(mod + block.shiftTable!.asType(mod.dtype))
        let normed = block.norm1(x) * (1 + m[1]) + m[0]; eval(normed)
        let qkv = block.attn.qkv(normed).reshaped(1, l, 3, h, d); eval(qkv)
        let q = qkv[0..., 0..., 0], k = qkv[0..., 0..., 1], v = qkv[0..., 0..., 2]
        let qn = block.attn.qNorm(applyInterleavedRotary(q, cos: cosA, sin: sinA)); eval(qn)
        let o = attend(qn, qn, v); eval(o)
        var rows: [(String, Double)] = []
        rows.append(("RePo (fp32 norm + 3 maps + cos/sin)", median("repo") { let r = rope(x); return r.0 + r.1 }))
        rows.append(("LayerNorm32 + modulation", median("norm+mod") { block.norm1(x) * (1 + m[1]) + m[0] }))
        rows.append(("GEMM qkv 1024→3072", median("qkv") { block.attn.qkv(normed) }))
        rows.append(("rotary + RMSNorm (q and k)", median("rope+rms") {
            block.attn.qNorm(applyInterleavedRotary(q, cos: cosA, sin: sinA)) + block.attn.kNorm(applyInterleavedRotary(k, cos: cosA, sin: sinA))
        }))
        rows.append(("SDPA 16 heads × 64 (fused)", median("sdpa") { attend(qn, qn, v) }))
        rows.append(("GEMM out 1024→1024", median("out") { block.attn.out(o) }))
        rows.append(("LayerNorm32 + MLP 1024→4096→1024", median("mlp") { block.mlp(block.norm2(x) * (1 + m[4]) + m[3]) }))
        rows.append(("whole block (RePo + block)", median("block") { let r = rope(x); return block(x, mod: mod, cos: r.0, sin: r.1) }))
        return rows
    }
}
