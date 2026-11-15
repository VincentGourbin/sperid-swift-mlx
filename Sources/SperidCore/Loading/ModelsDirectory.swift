// ModelsDirectory.swift - where weights live on this machine (never hard-coded in the repo)
// Copyright 2026 Vincent Gourbin

import Foundation

public enum ModelsDirectory {
    /// The environment variable that overrides the default location.
    public static let environmentKey = "SPERID_MODELS_DIR"

    /// `$SPERID_MODELS_DIR`, else `~/Library/Caches/models` (the cache the sibling ports share).
    public static func resolve(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        if let path = environment[environmentKey], !path.isEmpty {
            return URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
        }
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return caches.appendingPathComponent("models", isDirectory: true)
    }
}
