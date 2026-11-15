// IdolModels.swift - which weights are present, and getting the missing ones
// Copyright 2026 Vincent Gourbin

import Foundation
import SperidCore

/// The weights Idol reads: `speridlabs/idol` (Apache-2.0, 3.1 GB), the idol-turbo LoRAs (117 MB), and two gated
/// encoders — DINOv3 ViT-H+/16 (Meta's DINOv3 licence) and the FLUX.2 VAE (FLUX.2-dev terms). The gated ones need
/// the terms accepted on huggingface.co and a token (`HF_TOKEN`, or passed here).
public enum IdolModels {
    public struct Status: Sendable {
        /// Files still to download (encoders already in the Hugging Face cache count as present).
        public let missing: [HubFile]
        public var isComplete: Bool { missing.isEmpty }
        public var missingBytes: Int64 { missing.reduce(0) { $0 + $1.size } }
    }

    public static func files(includeTurbo: Bool = true) -> [HubFile] {
        IdolWeights.idolFiles + (includeTurbo ? [IdolWeights.turboLoRA] : []) + IdolWeights.encoderFiles
    }

    public static func status(modelsDirectory: URL = ModelsDirectory.resolve(), includeTurbo: Bool = true) -> Status {
        Status(missing: files(includeTurbo: includeTurbo).filter { file in
            if FileManager.default.fileExists(atPath: file.localURL(in: modelsDirectory).path) { return false }
            return EncoderWeights.locate(repo: file.repoID, file: file.path, modelsDirectory: modelsDirectory) == nil
        })
    }

    /// Downloads what `status` reports missing (resumable, SHA-256 checked). A gated file the account cannot
    /// read throws `SperidError.gatedRepository` with the page where the terms are accepted.
    public static func download(modelsDirectory: URL = ModelsDirectory.resolve(), includeTurbo: Bool = true,
                                token: String? = HubToken.resolve(),
                                progress: @Sendable @escaping (DownloadProgress) -> Void = { _ in }) async throws {
        let missing = status(modelsDirectory: modelsDirectory, includeTurbo: includeTurbo).missing
        guard !missing.isEmpty else { return }
        try await HubDownloader(root: modelsDirectory, token: token).download(missing, progress: progress)
    }
}
