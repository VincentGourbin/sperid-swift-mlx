// HubDownloader.swift - resumable, verified downloads of pinned Hugging Face files
// Copyright 2026 Vincent Gourbin
// Adapted from yue2-mlx-swift `ModelDownloader` (`.part` + HTTP Range resume, SHA-256 check).

import CryptoKit
import Foundation

/// Snapshot of one file's progress within a `HubDownloader.download` call.
public struct DownloadProgress: Sendable, Equatable {
    public let file: String
    public let fileIndex: Int
    public let fileCount: Int
    public let writtenBytes: Int64
    public let totalBytes: Int64
}

/// Downloads `HubFile`s into `<root>/<repo>/<path>`. Each file streams to `<name>.part` with HTTP
/// `Range` resume, is checked against its pinned SHA-256 (when it has one), then renamed into place.
/// A present file whose SHA-256 matches is skipped; a present file that does not match is replaced.
public actor HubDownloader {
    private let root: URL
    private let token: String?
    private let configuration: URLSessionConfiguration

    /// `configuration` is a test seam (`protocolClasses` for a stub server).
    public init(root: URL, token: String? = HubToken.resolve(),
                configuration: URLSessionConfiguration = .default) {
        self.root = root
        self.token = token
        self.configuration = configuration
    }

    public func download(_ files: [HubFile],
                         progress: @Sendable @escaping (DownloadProgress) -> Void = { _ in }) async throws {
        for (index, file) in files.enumerated() {
            let destination = file.localURL(in: root)
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            if try Self.isComplete(file, at: destination) { continue }
            try? FileManager.default.removeItem(at: destination)
            try await downloadOne(file, to: destination) { written, total in
                progress(DownloadProgress(file: file.path, fileIndex: index, fileCount: files.count,
                                          writtenBytes: written, totalBytes: total))
            }
            if let expected = file.sha256, try Self.sha256Hex(of: destination) != expected.lowercased() {
                try? FileManager.default.removeItem(at: destination)
                throw SperidError.weightMismatch("\(file.repoID)/\(file.path): SHA-256 differs from the pinned value (removed)")
            }
        }
    }

    /// Present with the pinned size, and the pinned SHA-256 when there is one.
    public static func isComplete(_ file: HubFile, at url: URL) throws -> Bool {
        guard let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int64,
              size == file.size else { return false }
        guard let expected = file.sha256 else { return true }
        return try sha256Hex(of: url) == expected.lowercased()
    }

    private func downloadOne(_ file: HubFile, to destination: URL,
                             progress: @Sendable @escaping (Int64, Int64) -> Void) async throws {
        let partURL = destination.appendingPathExtension("part")
        var request = URLRequest(url: file.url)
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let offset = ((try? FileManager.default.attributesOfItem(atPath: partURL.path))?[.size] as? Int64) ?? 0
        if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }

        let delegate = ChunkedDownloadDelegate(file: file, partURL: partURL, resumeOffset: offset, progress: progress)
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            delegate.completion = { continuation.resume(with: $0) }
            session.dataTask(with: request).resume()
        }
        try FileManager.default.moveItem(at: partURL, to: destination)
    }

    public static func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
