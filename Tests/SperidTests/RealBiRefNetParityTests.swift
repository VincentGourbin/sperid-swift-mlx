// RealBiRefNetParityTests.swift - BiRefNet on a Space example against the upstream model (tier 2)
// Copyright 2026 Vincent Gourbin
// Needs $SPERID_MODELS_DIR/parity/birefnet_food-avocado.safetensors (Scripts/reference/birefnet_fixtures.py).

import Foundation
import MLX
import Testing
@testable import IdolCore
@testable import SperidCore

enum RealBiRefNetFixture {
    static let url = RealFixtures.root.appendingPathComponent("parity/birefnet_food-avocado.safetensors")
    static var available: Bool { FileManager.default.fileExists(atPath: url.path) }
    nonisolated(unsafe) static let arrays: [String: MLXArray] = (try? loadArrays(url: url)) ?? [:]
}

@Suite("RealBiRefNetParity", .serialized, .enabled(if: RealBiRefNetFixture.available))
struct RealBiRefNetParityTests {
    static func nhwc(_ a: MLXArray) -> MLXArray { a.transposed(0, 2, 3, 1) }
    static func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        (sqrt(sum(square(a.asType(.float32) - b))) / sqrt(sum(square(b)))).item(Float.self)
    }

    @Test(arguments: [DType.float32, DType.float16])
    func matchesUpstream(dtype: DType) throws {
        let model = try BiRefNet.load(url: IdolWeights.birefnet.localURL(in: RealFixtures.root), computeDType: dtype)
        let f = RealBiRefNetFixture.arrays
        let x = Self.nhwc(f["input"]!).asType(dtype)
        let feats = model.encode(x)
        var report: [String] = []
        for (i, name) in ["x1", "x2", "x3", "x4"].enumerated() {
            report.append("\(name) \(Self.relL2(feats[i], Self.nhwc(f[name]!)))")
        }
        let alpha = model(x)
        let ra = Self.relL2(alpha, Self.nhwc(f["alpha1024"]!))
        let image = f["image"]!                                               // [H, W, 3] uint8
        let (h, w) = (image.dim(0), image.dim(1))
        let rgba = concatenated([image, MLXArray.full([h, w, 1], values: MLXArray(UInt8(255)))], axis: -1)
        let u8 = model.alpha(for: RGBAImage(width: w, height: h, pixels: rgba.reshaped(-1).asArray(UInt8.self)))
        let ref = f["alpha_uint8"]!.reshaped(-1).asArray(UInt8.self)
        let diffs = zip(u8, ref).map { abs(Int($0) - Int($1)) }
        let mean = Double(diffs.reduce(0, +)) / Double(diffs.count)
        let iou = { () -> Double in
            let a = u8.map { $0 > 127 }, b = ref.map { $0 > 127 }
            let i = zip(a, b).filter { $0 && $1 }.count, u = zip(a, b).filter { $0 || $1 }.count
            return Double(i) / Double(max(u, 1))
        }()
        print("REALBIREF \(dtype) \(report.joined(separator: ", ")) alpha1024 \(ra) | uint8 mean |Δ| \(mean), max \(diffs.max()!), IoU(>127) \(iou)")
        if dtype == .float32 { #expect(ra < 1e-3) }
        #expect(iou >= 0.99 && mean <= 1.0)
    }
}
