// ExportCommand.swift - `idol export`: a 3DGS .ply → mesh .glb / .usdz, printable .stl
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import IdolCore

struct ExportCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "export", abstract: "Extract a mesh from a splat (.ply): coloured .glb / .usdz, printable .stl.")

    @Option(help: "Input 3DGS .ply.") var ply: String
    @Option(help: "Output .glb (Blender, three.js, Unity…).") var glb: String?
    @Option(help: "Output .usdz (Quick Look, AR, Reality Composer).") var usdz: String?
    @Option(help: "Grid resolution (cells per axis).") var resolution = 192
    @Option(help: "Iso-level, fraction of the 99th percentile of the density.") var iso: Float = 0.04
    @OptionGroup var print3D: PrintArguments

    func run() throws {
        guard glb != nil || usdz != nil || print3D.stl != nil else { throw ValidationError("give --glb, --usdz and/or --stl") }
        let start = Date()
        let cloud = try PLYReader.read(URL(fileURLWithPath: ply))
        if let stl = print3D.stl { try print3D.write(cloud, to: URL(fileURLWithPath: stl)) }
        guard glb != nil || usdz != nil else { return }
        let mesh = MeshExtraction.extract(cloud, options: .init(resolution: resolution, isoFraction: iso))
        print(String(format: "mesh: %d vertices, %d triangles (%.1f s)", mesh.positions.count, mesh.triangleCount, Date().timeIntervalSince(start)))
        if let glb { try GLBWriter.write(mesh, to: URL(fileURLWithPath: glb)); print("GLB → \(glb)") }
        if let usdz { try USDZWriter.write(mesh, to: URL(fileURLWithPath: usdz)); print("USDZ → \(usdz)") }
        print(String(format: "EXPORT OK in %.1f s", Date().timeIntervalSince(start)))
    }
}

/// `--stl` and its print settings, shared by `export` and `generate`.
struct PrintArguments: ParsableArguments {
    @Option(help: "Output .stl for a 3D printer (closed, solid, one piece, millimetres).") var stl: String?
    @Option(help: "Printed height in mm.") var height: Float = 100
    @Option(help: "Minimum wall / part thickness in mm.") var thickness: Float = 1.2
    @Option(help: "Base: flat (cut flat) | plinth (slab under the object) | none.") var base = "flat"
    @Option(help: "Maximum triangles (quadric decimation; 0: none).") var maxTriangles = 200_000
    @Option(help: "Clean-up grid cells over the object's height (finer pits the surface).") var cells: Int?
    @Option(help: "Iso-level of the print solid, fraction of the 99th percentile of the density.") var printIso: Float?
    @Option(help: "Gaussians below this opacity are ignored for the print solid.") var printMinOpacity: Float?

    static var defaults: PrintArguments { try! PrintArguments.parse([]) }

    var options: PrintOptions {
        get throws {
            let b: PrintOptions.Base
            switch base {
            case "flat": b = .flat
            case "plinth": b = .plinth
            case "none": b = .none
            default: throw ValidationError("--base flat | plinth | none")
            }
            var o = PrintOptions(heightMM: height, minThicknessMM: thickness, base: b, maxTriangles: maxTriangles)
            if let cells { o.gridCells = cells }
            if let printIso { o.isoFraction = printIso }
            if let printMinOpacity { o.minimumOpacity = printMinOpacity }
            return o
        }
    }

    func write(_ cloud: GaussianCloud, to url: URL) throws {
        let start = Date()
        guard let mesh = PrintPreparation.prepare(cloud, options: try options) else { throw ValidationError("no matter left to print") }
        try STLWriter.write(mesh, to: url)
        print(String(format: "STL → %@ (%.1f s): %@", url.path, Date().timeIntervalSince(start), mesh.report.description))
    }
}
