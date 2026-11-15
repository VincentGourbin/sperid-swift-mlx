// ContentView.swift - Idol Studio's window, laid out like the speridlabs/idol Space
// Copyright 2026 Vincent Gourbin

import AppKit
import IdolCore
import IdolUI
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Bindable var model: StudioModel
    @State private var settingsOpen = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            HStack(alignment: .top, spacing: 20) {
                inputColumn.frame(width: 380)
                VStack(alignment: .leading, spacing: 8) {
                    Group {
                        if #available(macOS 27.0, *) {
                            SplatView(cloud: model.cloud, id: model.cloudVersion)
                        } else {
                            Text("The splat viewer needs macOS 27 (RealityKit Gaussian splats)").foregroundStyle(.secondary)
                        }
                    }
                    .frame(minHeight: 520)
                        .overlay(alignment: .center) {
                            if model.cloud == nil { Text("The splat appears here").foregroundStyle(.secondary) }
                        }
                    Text(model.timings).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                }
            }
            HStack {
                Spacer()
                Button("Quaternion order: \(model.quaternionOrderLabel)") { model.flipQuaternionOrder() }
                    .buttonStyle(.link).font(.caption)
            }
            Text("EXAMPLES — CLICK TO LOAD THE IMAGE AND ITS SETTINGS").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
            ScrollView(.horizontal) {
                LazyHStack(spacing: 8) {
                    ForEach(model.presets) { p in
                        Button { model.pick(p) } label: {
                            Group {
                                if let img = model.thumbnails[p.id] { Image(nsImage: img).resizable().scaledToFit() }
                                else { Color.gray.opacity(0.2) }
                            }
                            .frame(width: 96, height: 96)
                        }
                        .buttonStyle(.plain)
                        .help(p.title)
                    }
                }
            }
            .frame(height: 104)
        }
        .padding(24)
        .frame(minWidth: 1100, minHeight: 820)
    }

    var header: some View {
        HStack(alignment: .lastTextBaseline) {
            VStack(alignment: .leading, spacing: 6) {
                Text("SPERIDLABS · IDOL · SWIFT / MLX").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                Text("Idol Studio").font(.system(size: 44, weight: .bold))
                Text("One image in, a 3D object out — entirely on this Mac.").foregroundStyle(.secondary)
            }
            Spacer()
            Text(model.status).font(.callout).foregroundStyle(.secondary)
        }
    }

    var inputColumn: some View {
        VStack(alignment: .leading, spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 4).fill(Color.gray.opacity(0.12))
                if let img = model.inputImage { Image(nsImage: img).resizable().scaledToFit().padding(8) }
                else { Text("Drop an image, or click to choose").foregroundStyle(.secondary) }
            }
            .frame(height: 320)
            .onTapGesture { chooseImage() }
            .onDrop(of: [.fileURL], isTargeted: nil) { providers in
                _ = providers.first?.loadObject(ofClass: URL.self) { url, _ in
                    if let url { Task { @MainActor in model.open(url) } }
                }
                return true
            }
            Toggle("Turbo — 5 steps, about 20× faster (idol-turbo)", isOn: $model.turbo)
            Picker("Input type", selection: $model.mode) {
                Text("Synthetic").tag(IdolCropMode.synthetic)
                Text("Real photo").tag(IdolCropMode.real)
            }
            .pickerStyle(.segmented)
            HStack {
                Button(model.busy ? "Generating…" : "Generate") { model.generate() }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                    .disabled(!model.ready || model.busy || model.inputURL == nil)
                if model.busy { Button("Cancel") { model.cancel() } }
                Spacer()
                Button("Open .ply…") {
                    let panel = NSOpenPanel()
                    panel.allowedContentTypes = [UTType(filenameExtension: "ply") ?? .data]
                    if panel.runModal() == .OK, let url = panel.url { model.openPLY(url) }
                }
                if model.cloud != nil {
                    Menu("Export") {
                        Button("USDZ (Quick Look, AR)…") { saveMesh("usdz") }
                        Button("GLB (Blender, web)…") { saveMesh("glb") }
                    }
                    .fixedSize()
                }
                if let folder = model.lastOutputFolder {
                    Button("Show .ply") { NSWorkspace.shared.activateFileViewerSelecting([folder]) }
                }
            }
            if model.busy, let stage = model.stage {
                ProgressView(value: stage == .denoising ? model.fraction : nil) { Text(stage.rawValue.capitalized) }
            }
            DisclosureGroup("Settings", isExpanded: $settingsOpen) {
                Picker("Gaussians", selection: $model.numGaussians) {
                    ForEach(StudioModel.gaussianChoices, id: \.self) { Text($0.formatted()).tag($0) }
                }
                slider("Steps", value: Binding(get: { Double(model.steps) }, set: { model.steps = Int($0) }), 10...100, step: 1, format: "%.0f")
                slider("Guidance scale", value: $model.cfg, 0...10, step: 0.1, format: "%.1f")
                slider("Time shift", value: $model.rescaleT, 1...6, step: 0.1, format: "%.1f")
                HStack {
                    Text("Seed")
                    TextField("Seed", value: $model.seed, format: .number).frame(width: 120)
                    Toggle("Random seed", isOn: $model.randomSeed)
                }
            }
        }
    }

    func slider(_ title: String, value: Binding<Double>, _ range: ClosedRange<Double>, step: Double, format: String) -> some View {
        HStack {
            Text(title).frame(width: 110, alignment: .leading)
            Slider(value: value, in: range, step: step)
            Text(String(format: format, value.wrappedValue)).monospacedDigit().frame(width: 44, alignment: .trailing)
        }
    }

    func saveMesh(_ ext: String) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(filenameExtension: ext) ?? .data]
        panel.nameFieldStringValue = "idol.\(ext)"
        if panel.runModal() == .OK, let url = panel.url { model.exportMesh(to: url) }
    }

    func chooseImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK, let url = panel.url { model.open(url) }
    }
}
