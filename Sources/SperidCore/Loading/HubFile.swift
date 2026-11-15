// HubFile.swift - one pinned Hugging Face file (repo, revision, size, SHA-256)
// Copyright 2026 Vincent Gourbin

import Foundation

/// A weight file as published on the Hub, pinned to a revision. `sha256` is the LFS object id
/// (`lfs.oid` in the Hub tree API), `nil` for small non-LFS files (configs).
public struct HubFile: Sendable, Equatable, Hashable {
    public let repoID: String
    public let revision: String
    public let path: String
    public let size: Int64
    public let sha256: String?
    /// The repository needs accepted terms (DINOv3, FLUX.2-dev): downloads need `HF_TOKEN`.
    public let gated: Bool

    public init(repoID: String, revision: String, path: String, size: Int64, sha256: String?, gated: Bool = false) {
        self.repoID = repoID
        self.revision = revision
        self.path = path
        self.size = size
        self.sha256 = sha256
        self.gated = gated
    }

    /// `https://huggingface.co/<repo>/resolve/<revision>/<path>`.
    public var url: URL {
        URL(string: "https://huggingface.co/\(repoID)/resolve/\(revision)/\(path)")!
    }

    /// Where the file lives under the models directory: `<root>/<repo>/<path>`.
    public func localURL(in root: URL) -> URL {
        root.appendingPathComponent(repoID).appendingPathComponent(path)
    }
}
