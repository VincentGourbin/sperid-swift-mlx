// HubToken.swift - the Hugging Face token a user already has
// Copyright 2026 Vincent Gourbin

import Foundation

public enum HubToken {
    /// `HF_TOKEN`, else the file `hf auth login` writes (`$HF_HOME/token`, default `~/.cache/huggingface/token`).
    public static func resolve(environment: [String: String] = ProcessInfo.processInfo.environment) -> String? {
        if let t = environment["HF_TOKEN"], !t.isEmpty { return t }
        let home = environment["HF_HOME"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface")
        let token = try? String(contentsOf: home.appendingPathComponent("token"), encoding: .utf8)
        return token?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
    }
}

extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
