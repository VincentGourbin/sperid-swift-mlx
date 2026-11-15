// DownloadCommand.swift - `idol download`: every weight Idol reads (resumable, SHA-256 checked)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import IdolCore
import SperidCore

struct DownloadCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download",
        abstract: "Download the pinned weights: speridlabs/idol, idol-turbo, and the gated DINOv3 and FLUX.2 VAE (hf auth login).")

    @Option(help: "Models directory (default: $SPERID_MODELS_DIR, else ~/Library/Caches/models).")
    var modelsDir: String?

    @Flag(help: "Skip the idol-turbo LoRAs (117 MB).") var noTurbo = false

    func run() async throws {
        let root = modelsDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? ModelsDirectory.resolve()
        let status = IdolModels.status(modelsDirectory: root, includeTurbo: !noTurbo)
        guard !status.isComplete else { print("DOWNLOAD OK (everything present in \(root.path))"); return }
        print(String(format: "downloading %d files (%.2f GB) into %@", status.missing.count, Double(status.missingBytes) / 1e9, root.path))
        let reporter = ProgressReporter()
        try await IdolModels.download(modelsDirectory: root, includeTurbo: !noTurbo) { reporter.report($0) }
        print("DOWNLOAD OK (\(status.missing.count) files)")
    }
}

/// Prints one line per file at every 10 % (stdout stays readable in logs).
final class ProgressReporter: @unchecked Sendable {
    private var lastDecile: [String: Int] = [:]
    private let lock = NSLock()

    func report(_ p: DownloadProgress) {
        guard p.totalBytes > 0 else { return }
        let decile = Int(10 * p.writtenBytes / p.totalBytes)
        lock.lock(); defer { lock.unlock() }
        guard decile != lastDecile[p.file] else { return }
        lastDecile[p.file] = decile
        print(String(format: "  [%d/%d] %@ %3d %%", p.fileIndex + 1, p.fileCount, p.file, decile * 10))
    }
}
