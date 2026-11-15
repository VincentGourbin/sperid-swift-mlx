// IdolStudioApp.swift - Idol Studio: the speridlabs/idol Space, locally in Swift / MLX / RealityKit
// Copyright 2026 Vincent Gourbin

import AppKit
import SwiftUI

@main
struct IdolStudioApp: App {
    @State private var model = StudioModel()

    init() {
        // launched from the command line as an SPM executable: make it a regular, focused app
        NSApplication.shared.setActivationPolicy(.regular)
        DispatchQueue.main.async { NSApplication.shared.activate(ignoringOtherApps: true) }
    }

    var body: some Scene {
        WindowGroup("Idol Studio") {
            ContentView(model: model).onAppear { model.start() }
        }
        .windowResizability(.contentMinSize)
    }
}
