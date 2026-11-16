// IdolResult.swift - a generated object and its outputs (PLY, USDZ, GLB)
// Copyright 2026 Vincent Gourbin

import CoreGraphics
import Foundation
import SperidCore

/// Everything a generation produced. Plain values: safe to pass between tasks and actors.
public struct IdolResult: Sendable {
    /// The 3D Gaussians (activated values, object z-up in [-0.5, 0.5]³).
    public let cloud: GaussianCloud
    /// The 1024² image the model saw (background removed, cropped, on black).
    public let preprocessed: RGBAImage
    /// The camera the model predicted for the input image.
    public let camera: CameraPose
    public let settings: IdolSettings
    /// The seed actually used (drawn when `settings.seed` was nil).
    public let seed: UInt64
    /// Seconds per stage: preprocess, encode, denoise, decode.
    public let timings: [String: Double]

    /// Standard 3DGS binary PLY (what SuperSplat, Blender 5.3, the SDK read).
    public func plyData() -> Data { PLYWriter.data(cloud) }
    public func writePLY(to url: URL) throws { try PLYWriter.write(cloud, to: url) }

    /// A coloured mesh of the object (density grid + surface nets, y-up, about 1 m).
    public func mesh(options: MeshExtractionOptions = .init()) -> SplatMesh { MeshExtraction.extract(cloud, options: options) }
    /// Textured mesh for AR Quick Look, RealityKit, Reality Composer (passes `usdchecker --arkit`).
    public func writeUSDZ(to url: URL, options: MeshExtractionOptions = .init()) throws { try USDZWriter.write(mesh(options: options), to: url) }
    /// Mesh with vertex colours for Blender, three.js, Unity.
    public func writeGLB(to url: URL, options: MeshExtractionOptions = .init()) throws { try GLBWriter.write(mesh(options: options), to: url) }

    /// A closed, solid, single-piece mesh in millimetres for a 3D printer (see `PrintOptions`), or nil when no
    /// matter survives the clean-up. Check `report.isPrintable`.
    public func printMesh(options: PrintOptions = .init()) -> PrintMesh? { PrintPreparation.prepare(cloud, options: options) }
    /// Binary STL for a slicer (millimetres, z up, standing on the bed). Returns the checks made on it.
    @discardableResult
    public func writeSTL(to url: URL, options: PrintOptions = .init()) throws -> PrintReport {
        guard let mesh = printMesh(options: options) else { throw SperidError.invalidRequest("no matter left to print") }
        try STLWriter.write(mesh, to: url)
        return mesh.report
    }

    /// `<name>.ply`, `<name>_input.png`, `<name>_camera.json`, like `python -m idol.sample`.
    public func write(to folder: URL, name: String) throws {
        try IdolOutput(cloud: cloud, preprocessed: preprocessed, camera: camera, timings: timings).write(to: folder, name: name)
    }

    public var preprocessedCGImage: CGImage { preprocessed.cgImage(premultipliedOnBlack: true) }
}
