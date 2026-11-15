// SystematicSampler.swift - split n points among 8 octree children (`decoder.py`, CPU)
// Copyright 2026 Vincent Gourbin

import Foundation

/// `systematic_sample` of `decoder.py`, in plain Swift: the octree descent has data-dependent
/// shapes (the number of occupied cells changes per level), which suits the CPU better than an
/// MLX graph. Only the 8 model forwards (one per level) run on the GPU.
public enum SystematicSampler {
    /// Splits `count` items among the bins of `probabilities` by systematic resampling.
    /// - Parameters:
    ///   - probabilities: one row of P bin weights (clamped at 0; a zero row becomes uniform).
    ///     u is non-decreasing in k, so one forward scan replaces `searchsorted`.
    ///   - count: items to distribute (sum of the result).
    ///   - offset: u0 ∈ [0, 1) *before* scaling; the stratified points are (u0 + k)/n.
    /// - Returns: P counts summing to `count`.
    public static func split(probabilities: [Float], count: Int, offset: Float) -> [Int] {
        let bins = probabilities.count
        var result = [Int](repeating: 0, count: bins)
        guard count > 0, bins > 0 else { return result }
        var p = probabilities.map { max(0, $0) }
        let sum = p.reduce(0, +)
        if sum == 0 {
            p = [Float](repeating: 1 / Float(bins), count: bins)
        } else {
            // the reference divides by clamp_min(sum, 1): a row summing below 1 is NOT rescaled up
            p = p.map { $0 / max(sum, 1) }
        }
        // cumulative distribution; the reference clamps it at 1 − 1e-12, which is 1.0 in fp32
        var cdf = [Float](repeating: 0, count: bins)
        var running: Float = 0
        for i in 0..<bins {
            running += p[i]
            cdf[i] = min(running, 1)
        }
        let n = Float(count)
        let u0 = offset / n
        var bin = 0
        for k in 0..<count {
            let u = min(u0 + Float(k) / n, 1)
            // searchsorted(side='left'), clamped to P − 1: first index with cdf[index] >= u
            while bin < bins - 1 && cdf[bin] < u { bin += 1 }
            result[bin] += 1
        }
        return result
    }
}
