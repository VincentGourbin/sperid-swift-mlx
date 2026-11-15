// TinyBiRefNetOpsTests.swift - deformable conv and align-corners resize against torch (tier 1)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import IdolCore

@Suite("TinyBiRefNetOps", .serialized)
struct TinyBiRefNetOpsTests {
    nonisolated(unsafe) static let arrays: [String: MLXArray] = (try? loadArrays(url: TinyParity.repoRoot.appendingPathComponent("parity/tiny_birefnet.safetensors"))) ?? [:]
    static func nhwc(_ a: MLXArray) -> MLXArray { a.transposed(0, 2, 3, 1) }

    @Test(arguments: [1, 3, 7])
    func deformConvMatchesTorchvision(k: Int) {
        let f = Self.arrays
        let y = DeformConv.apply(Self.nhwc(f["deform\(k).x"]!), offset: Self.nhwc(f["deform\(k).offset"]!),
                                 mask: Self.nhwc(f["deform\(k).mask"]!), weight: f["deform\(k).w"]!.transposed(0, 2, 3, 1),
                                 bias: nil, kernel: k, padding: k / 2)
        let r = TinyParity.relMax(y.transposed(0, 3, 1, 2), f["deform\(k).y"]!)
        print("TINYDEFORM k=\(k) relMax \(r)")
        #expect(r < 1e-5)
    }

    @Test func resizeMatchesInterpolate() {
        let f = Self.arrays
        let x = Self.nhwc(f["resize.x"]!)
        for (name, h, w, src) in [("up", 16, 20, x), ("down", 3, 5, x), ("one", 4, 6, x[0..., 0..<1, 0..<1])] {
            let y = Resample.bilinear(src, height: h, width: w).transposed(0, 3, 1, 2)
            #expect(TinyParity.relMax(y, f["resize.\(name)"]!) < 1e-5, "\(name)")  // fp32 rounding order: 1.2e-6
        }
    }
}
