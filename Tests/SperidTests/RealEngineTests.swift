// RealEngineTests.swift - the public API end to end, as an app uses it (tier 2, ~1 min)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import IdolCore
@testable import SperidCore

enum EngineFixture {
    static let image = RealFixtures.root.appendingPathComponent("speridlabs-idol-space/examples/fig-fox.webp")
    static var available: Bool {
        IdolModels.status(modelsDirectory: RealFixtures.root).isComplete && FileManager.default.fileExists(atPath: image.path)
    }
}

@Suite("RealEngine", .serialized, .enabled(if: EngineFixture.available))
struct RealEngineTests {
    @Test func generatesAndExports() async throws {
        let engine = try await IdolEngine.load(modelsDirectory: RealFixtures.root)
        #expect(engine.hasTurbo)
        let stages = StageLog()
        let result = try await engine.generate(.url(EngineFixture.image), settings: .turbo) { stages.add($0.stage) }
        #expect(result.cloud.count == 262_144)
        #expect(result.seed == 42)
        #expect(stages.all.contains(.denoising) && stages.all.last == .done)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("idol-engine-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try result.writePLY(to: dir.appendingPathComponent("fox.ply"))
        try result.writeUSDZ(to: dir.appendingPathComponent("fox.usdz"))
        try result.writeGLB(to: dir.appendingPathComponent("fox.glb"))
        #expect(try PLYReader.read(dir.appendingPathComponent("fox.ply")).count == 262_144)
        print("REALENGINE timings \(result.timings.mapValues { String(format: "%.1f", $0) })")
    }

    @Test func cancellationStopsAGeneration() async throws {
        let engine = try await IdolEngine.load(modelsDirectory: RealFixtures.root)
        let task = Task { try await engine.generate(.url(EngineFixture.image), settings: .quality) }
        try await Task.sleep(for: .seconds(8))
        let start = Date()
        task.cancel()
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        print(String(format: "REALENGINE cancelled after %.1f s", Date().timeIntervalSince(start)))
        #expect(Date().timeIntervalSince(start) < 20)
    }
}

final class StageLog: @unchecked Sendable {
    private let lock = NSLock()
    private var stages: [IdolStage] = []
    func add(_ s: IdolStage) { lock.lock(); if stages.last != s { stages.append(s) }; lock.unlock() }
    var all: [IdolStage] { lock.lock(); defer { lock.unlock() }; return stages }
}
