// ExportCommand.swift - `idol export`: a 3DGS .ply → mesh .glb / .usdz
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import IdolCore

struct ExportCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "export", abstract: "Extract a coloured mesh from a splat (.ply) and write .glb and/or .usdz.")

    @Option(help: "Input 3DGS .ply.") var ply: String
    @Option(help: "Output .glb (Blender, three.js, Unity…).") var glb: String?
    @Option(help: "Output .usdz (Quick Look, AR, Reality Composer).") var usdz: String?
    @Option(help: "Grid resolution (cells per axis).") var resolution = 192
    @Option(help: "Iso-level, fraction of the 99th percentile of the density.") var iso: Float = 0.04

    func run() throws {
        guard glb != nil || usdz != nil else { throw ValidationError("give --glb and/or --usdz") }
        let start = Date()
        let cloud = try PLYReader.read(URL(fileURLWithPath: ply))
        let mesh = MeshExtraction.extract(cloud, options: .init(resolution: resolution, isoFraction: iso))
        print(String(format: "mesh: %d vertices, %d triangles (%.1f s)", mesh.positions.count, mesh.triangleCount, Date().timeIntervalSince(start)))
        if let glb { try GLBWriter.write(mesh, to: URL(fileURLWithPath: glb)); print("GLB → \(glb)") }
        if let usdz { try USDZWriter.write(mesh, to: URL(fileURLWithPath: usdz)); print("USDZ → \(usdz)") }
        print(String(format: "EXPORT OK in %.1f s", Date().timeIntervalSince(start)))
    }
}
