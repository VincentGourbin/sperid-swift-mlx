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

    /// Cancels in the middle of the denoising (not during the encoders): the latency an app sees. The check
    /// sits between steps and between the two forwards of a guided step, so the bound is half a step.
    @Test func cancellationStopsAGeneration() async throws {
        let engine = try await IdolEngine.load(modelsDirectory: RealFixtures.root)
        let steps = StepClock()
        let task = Task { try await engine.generate(.url(EngineFixture.image), settings: .quality) { p in
            if p.stage == .denoising { steps.tick() }
        } }
        #expect(try await waitUntil(.seconds(300)) { steps.count >= 3 })   // two full steps timed
        try await Task.sleep(for: .seconds(steps.longest / 2))             // cancel mid-step
        let start = Date()
        task.cancel()
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        let latency = Date().timeIntervalSince(start)
        print(String(format: "REALENGINE quality cancelled after %.1f s (guided step %.1f s)", latency, steps.longest))
        #expect(latency < steps.longest * 0.6 + 2)
    }

    /// Fluxforge drops the engine whenever its GPU queue moves to another model: each cycle must give its
    /// thread back (one leaked per cycle before 0.3.2).
    @Test func releaseAndReloadLeakNoThread() async throws {
        let before = processThreadCount()
        var engineThreads: [Int] = []
        for _ in 0..<3 {
            var engine: IdolEngine? = try await IdolEngine.load(modelsDirectory: RealFixtures.root)
            engineThreads.append(processThreadCount())
            _ = engine?.hasTurbo
            engine = nil
            #expect(try await waitUntil { processThreadCount() <= before + 2 })
        }
        let after = processThreadCount()
        print("REALENGINE threads before \(before), with engine \(engineThreads), after 3 cycles \(after)")
        #expect(after <= before + 2)
    }
}

final class StageLog: @unchecked Sendable {
    private let lock = NSLock()
    private var stages: [IdolStage] = []
    func add(_ s: IdolStage) { lock.lock(); if stages.last != s { stages.append(s) }; lock.unlock() }
    var all: [IdolStage] { lock.lock(); defer { lock.unlock() }; return stages }
}

final class StepClock: @unchecked Sendable {
    private let lock = NSLock()
    private var times: [Date] = []
    func tick() { lock.lock(); times.append(Date()); lock.unlock() }
    var count: Int { lock.lock(); defer { lock.unlock() }; return times.count }
    /// Longest interval between two denoising progress events (one step).
    var longest: Double {
        lock.lock(); defer { lock.unlock() }
        return zip(times.dropFirst(), times).map { $0.timeIntervalSince($1) }.max() ?? 0
    }
}
