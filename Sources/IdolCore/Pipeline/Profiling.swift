// Profiling.swift - phases for swift-mlx-profiler, free when no session is active
// Copyright 2026 Vincent Gourbin

import Foundation
import MLXProfiler

/// Every stage of the pipeline reports here; nothing happens unless a `ProfilingSession` is the active
/// session of `MLXProfiler.shared` (`idol generate --profile DIR`). Phases are evaluated inside their own
/// bounds (`eval` at the end of the body), so their duration is the work, not a lazy graph.
enum IdolProfiling {
    static var session: ProfilingSession? {
        MLXProfiler.shared.isEnabled ? MLXProfiler.shared.activeSession : nil
    }

    static func phase<T>(_ name: String, _ category: ProfilingCategory, _ body: () throws -> T) rethrows -> T {
        guard let s = session else { return try body() }
        s.beginPhase(name, category: category)
        defer { s.endPhase(name, category: category) }
        return try body()
    }

    static func step(_ index: Int, of total: Int, since start: UInt64) {
        guard let s = session else { return }
        s.recordStep(index: index, total: total, durationUs: (DispatchTime.now().uptimeNanoseconds - start) / 1000)
    }

    /// Accumulated sub-durations inside a phase (e.g. the octree's GPU forwards against its CPU split).
    static func complete(_ name: String, _ category: ProfilingCategory, microseconds: UInt64) {
        session?.recordComplete(name, category: category, durationUs: microseconds)
    }

    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
}
