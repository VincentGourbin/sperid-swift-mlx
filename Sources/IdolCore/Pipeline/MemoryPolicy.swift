// MemoryPolicy.swift - MLX buffer cache bounds for a generation (opt-in, restored)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// MLX keeps freed buffers in a cache that, unbounded, took the process to 34 GB for an 11 GB active peak
/// (`idol generate --profile`, 2026-11-15; pattern MLX-010). A library must not change process-wide
/// settings for its host by default: `apply()` is explicit, `restore()` gives the previous value back.
public struct IdolMemoryPolicy: Sendable {
    /// MLX `cacheLimit` in MB during the generation (`nil` = leave it as is).
    public var cacheLimitMB: Int?
    /// `Memory.clearCache()` after the encoders, the denoiser and the decoders.
    public var clearBetweenStages: Bool

    public init(cacheLimitMB: Int? = nil, clearBetweenStages: Bool = false) {
        self.cacheLimitMB = cacheLimitMB
        self.clearBetweenStages = clearBetweenStages
    }

    /// No change (the default).
    public static let unbounded = IdolMemoryPolicy()
    /// Bounded cache, cleared between stages.
    public static func bounded(cacheMB: Int = 2048) -> IdolMemoryPolicy { .init(cacheLimitMB: cacheMB, clearBetweenStages: true) }

    /// The CLI's and the app's choice: `bounded(2048)` (process peak 33.8 → 11.3 GB in turbo, time within
    /// noise, A/B/B/A 2026-11-15) unless `IDOL_CACHE_LIMIT_MB` says otherwise (`0` = unbounded).
    public static var fromEnvironment: IdolMemoryPolicy {
        guard let v = ProcessInfo.processInfo.environment["IDOL_CACHE_LIMIT_MB"], let mb = Int(v) else { return .bounded() }
        return mb > 0 ? .bounded(cacheMB: mb) : .unbounded
    }

    /// Returns the previous cache limit, to pass to `restore`.
    public func apply() -> Int? {
        guard let mb = cacheLimitMB else { return nil }
        let previous = Memory.cacheLimit
        Memory.cacheLimit = mb * 1_048_576
        return previous
    }

    public func restore(_ previous: Int?) {
        if let previous { Memory.cacheLimit = previous }
    }

    func stageDone() { if clearBetweenStages { Memory.clearCache() } }
}
