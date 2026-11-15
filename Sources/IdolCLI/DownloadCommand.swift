// DownloadCommand.swift - `idol download`: the pinned weights, resumable and verified
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import IdolCore
import SperidCore

struct DownloadCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download", abstract: "Download the pinned Idol weights (resume, SHA-256 check).")

    @Option(help: "Models directory (default: $SPERID_MODELS_DIR, else ~/Library/Caches/models).")
    var modelsDir: String?

    @Flag(help: "Also download the idol-turbo phase LoRAs (117 MB).")
    var turbo = false

    func run() async throws {
        let root = modelsDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
            ?? ModelsDirectory.resolve()
        let files = IdolWeights.idolFiles + (turbo ? [IdolWeights.turboLoRA] : [])
        print("downloading \(files.count) files into \(root.path)")
        let downloader = HubDownloader(root: root)
        let reporter = ProgressReporter()
        try await downloader.download(files) { reporter.report($0) }
        print("DOWNLOAD OK (\(files.count) files)")
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
