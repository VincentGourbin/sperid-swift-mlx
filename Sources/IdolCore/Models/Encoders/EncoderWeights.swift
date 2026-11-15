// EncoderWeights.swift - where the two image encoders live on this machine
// Copyright 2026 Vincent Gourbin

import Foundation
import SperidCore

/// The encoders come from the gated official repositories (accepted terms, HF_TOKEN), as the SDK reads them.
/// They are resolved in the models directory first (`<root>/<repo>/<path>`), then in the Hugging Face cache.
public enum EncoderWeights {
    public static let dinov3Repo = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
    public static let flux2Repo = "black-forest-labs/FLUX.2-dev"
    public static let dinov3File = "model.safetensors"
    public static let flux2VAEFile = "vae/diffusion_pytorch_model.safetensors"

    public static func locate(repo: String, file: String, modelsDirectory: URL,
                              environment: [String: String] = ProcessInfo.processInfo.environment) -> URL? {
        let direct = modelsDirectory.appendingPathComponent(repo).appendingPathComponent(file)
        if FileManager.default.fileExists(atPath: direct.path) { return direct }
        let hub = environment["HF_HUB_CACHE"].map { URL(fileURLWithPath: $0) }
            ?? environment["HF_HOME"].map { URL(fileURLWithPath: $0).appendingPathComponent("hub") }
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub")
        let snapshots = hub.appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots")
        let revisions = (try? FileManager.default.contentsOfDirectory(atPath: snapshots.path)) ?? []
        for revision in revisions.sorted() {
            let url = snapshots.appendingPathComponent(revision).appendingPathComponent(file)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    public static func dinov3(modelsDirectory: URL) throws -> URL {
        guard let url = locate(repo: dinov3Repo, file: dinov3File, modelsDirectory: modelsDirectory) else {
            throw SperidError.missingFile("\(dinov3Repo)/\(dinov3File) (gated: accept the terms, then `hf download \(dinov3Repo)`)")
        }
        return url
    }

    public static func flux2VAE(modelsDirectory: URL) throws -> URL {
        guard let url = locate(repo: flux2Repo, file: flux2VAEFile, modelsDirectory: modelsDirectory) else {
            throw SperidError.missingFile("\(flux2Repo)/\(flux2VAEFile) (gated: `hf download \(flux2Repo) --include 'vae/*'`)")
        }
        return url
    }
}
