// IdolStudioApp.swift - Idol Studio: the speridlabs/idol Space, locally in Swift / MLX / RealityKit
// Copyright 2026 Vincent Gourbin

import AppKit
import IdolCore
import IdolUI
import RealityKit
import SwiftUI

@main
struct IdolStudioApp: App {
    @State private var model = StudioModel()

    init() {
        // `idol-studio file.ply --render out.png [--wfirst]`: offscreen render, then quit (no window needed)
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--render"), i + 1 < args.count,
           let usdz = args.first(where: { $0.hasSuffix(".usdz") }) {
            if #available(macOS 27.0, *) {
                Task { @MainActor in
                    do {
                        let entity = try await Entity(contentsOf: URL(fileURLWithPath: usdz))
                        try await SplatRenderer.render(entity: entity, to: URL(fileURLWithPath: args[i + 1]), lit: true)
                        print("RENDER OK \(args[i + 1])"); exit(0)
                    } catch { print("RENDER FAILED \(error)"); exit(1) }
                }
            }
            return
        }
        if let i = args.firstIndex(of: "--render"), i + 1 < args.count,
           let ply = args.first(where: { $0.hasSuffix(".ply") }) {
            if #available(macOS 27.0, *) {
                SplatEntity.wFirst = args.contains("--wfirst")
                Task { @MainActor in
                    do {
                        let cameraURL = URL(fileURLWithPath: ply.replacingOccurrences(of: ".ply", with: "_camera.json"))
                        let pose = try? JSONDecoder().decode(CameraPose.self, from: Data(contentsOf: cameraURL))
                        try await SplatRenderer.render(try PLYReader.read(URL(fileURLWithPath: ply)),
                                                         to: URL(fileURLWithPath: args[i + 1]), inputCamera: pose)
                        print("RENDER OK \(args[i + 1])")
                        exit(0)
                    } catch { print("RENDER FAILED \(error)"); exit(1) }
                }
            }
            return
        }
        // launched from the command line as an SPM executable: make it a regular, focused app
        NSApplication.shared.setActivationPolicy(.regular)
        DispatchQueue.main.async { NSApplication.shared.activate(ignoringOtherApps: true) }
    }

    var body: some SwiftUI.Scene {
        WindowGroup("Idol Studio") {
            ContentView(model: model).onAppear { model.start() }
        }
        .windowResizability(.contentMinSize)
    }
}
