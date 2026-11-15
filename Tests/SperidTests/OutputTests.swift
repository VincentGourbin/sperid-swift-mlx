// OutputTests.swift - activations, PLY layout, camera decoding
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import IdolCore

@Suite("Output")
struct OutputTests {
    static func oneGaussian() -> GaussianCloud {
        GaussianCloud.activate(rawXYZ: [0.5, 0.5, 1.0], featuresDC: [0.1, 0.2, 0.3], rawScaling: [0, 0, 0],
                               rawRotation: [0, 0, 0, 0], rawOpacity: [0],
                               scalingBias: 0.004, opacityBias: 0.1, minimumKernelSize: 0.0009)
    }

    @Test func activationsUseTheBiases() {
        let g = Self.oneGaussian()
        #expect(g.xyz == [0, 0, 0.5])
        #expect(abs(g.opacity[0] - 0.1) < 1e-6)  // σ(σ⁻¹(0.1))
        let expectedScale = (0.004 * 0.004 + 0.0009 * 0.0009).squareRoot()
        #expect(abs(Double(g.scaling[0]) - expectedScale) < 1e-7)
        #expect(g.rotation == [1, 0, 0, 0])
    }

    @Test func plyHeaderIsTheStandard3DGSLayout() {
        let h = PLYWriter.header(vertexCount: 2)
        #expect(h.hasPrefix("ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\n"))
        #expect(h.hasSuffix("property float rot_3\nend_header\n"))
        #expect(PLYWriter.propertyNames.count == 17)
    }

    @Test func plyTurnsZUpIntoYUp() {
        let g = Self.oneGaussian()
        let v = PLYWriter.vertex(g, 0)
        #expect(v[0] == 0 && v[1] == -0.5 && v[2] == 0)        // (x, -z, y)
        let a = Float(0.5).squareRoot()
        #expect(abs(v[13] - a) < 1e-6 && abs(v[14] - a) < 1e-6)  // identity → 90° about x
        #expect(abs(v[9] - log(Float(0.1) / 0.9)) < 1e-5)       // logit opacity
        let data = PLYWriter.data(g)
        #expect(data.count == PLYWriter.header(vertexCount: 1).utf8.count + 17 * 4)
    }

    @Test func cameraMatchesThePythonFormula() {
        // decode_camera([0.3, 0.8, 0.2, 0.5, 0.4]) in Python
        let c = CameraPose(latent: [0.3, 0.8, 0.2, 0.5, 0.4])
        #expect(abs(c.yaw - 0.3587706702705722) < 1e-6)
        #expect(abs(c.pitch - 0.2299421051869875) < 1e-6)
        #expect(abs(c.radius - 2) < 1e-6 && abs(c.fov - 22.61986494804043) < 1e-5)
        let ortho = CameraPose(latent: [0, 0, 1, 0, 0.4])
        #expect(ortho.radius == 1000 && abs(ortho.fov - 0.04583662116584617) < 1e-6)
    }
}
