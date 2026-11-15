// ChunkedDownloadDelegate.swift - writes each received chunk to a resumable `.part` file
// Copyright 2026 Vincent Gourbin
// From yue2-mlx-swift `ModelDownloader.swift`; adds the gated-repository error.

import Foundation

/// Writing every chunk as it arrives (instead of `URLSessionDownloadTask`'s single temp file)
/// leaves a resumable `.part` after a dropped connection on a multi-gigabyte file.
final class ChunkedDownloadDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let file: HubFile
    private let partURL: URL
    private var handle: FileHandle?
    private var written: Int64
    private var total: Int64 = 0
    private let progress: @Sendable (Int64, Int64) -> Void
    var completion: ((Result<Void, Error>) -> Void)?

    init(file: HubFile, partURL: URL, resumeOffset: Int64, progress: @escaping @Sendable (Int64, Int64) -> Void) {
        self.file = file
        self.partURL = partURL
        self.written = resumeOffset
        self.progress = progress
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let status = (response as? HTTPURLResponse)?.statusCode ?? -1
        guard (200...299).contains(status), let http = response as? HTTPURLResponse else {
            let error: SperidError = (status == 401 || status == 403)
                ? .gatedRepository("HTTP \(status) for \(file.repoID)/\(file.path): accept the terms on "
                    + "https://huggingface.co/\(file.repoID) and set HF_TOKEN")
                : .missingFile("HTTP \(status) for \(file.repoID)/\(file.path)")
            completion?(.failure(error))
            completion = nil
            completionHandler(.cancel)
            return
        }
        let resumed = status == 206
        if !resumed {
            written = 0
            try? FileManager.default.removeItem(at: partURL)
        }
        total = http.value(forHTTPHeaderField: "Content-Length").flatMap(Int64.init).map { $0 + written } ?? 0
        if !FileManager.default.fileExists(atPath: partURL.path) {
            FileManager.default.createFile(atPath: partURL.path, contents: nil)
        }
        handle = try? FileHandle(forWritingTo: partURL)
        if resumed { _ = try? handle?.seekToEnd() }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let handle else { return }
        try? handle.write(contentsOf: data)
        written += Int64(data.count)
        progress(written, total)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        try? handle?.close()
        guard let completion else { return }
        self.completion = nil
        completion(error.map { .failure($0) } ?? .success(()))
    }
}
