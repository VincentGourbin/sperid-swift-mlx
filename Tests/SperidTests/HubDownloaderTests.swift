// HubDownloaderTests.swift - download, verification and skip logic against a stub server
// Copyright 2026 Vincent Gourbin

import CryptoKit
import Foundation
import Testing
@testable import SperidCore

/// Serves `StubServer.routes[url.path]` as (status, body); counts requests per path.
final class StubServer: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var routes: [String: (Int, Data)] = [:]
    nonisolated(unsafe) static var hits: [String: Int] = [:]
    static let lock = NSLock()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let path = request.url!.path
        Self.lock.lock()
        let (status, body) = Self.routes[path] ?? (404, Data())
        Self.hits[path, default: 0] += 1
        Self.lock.unlock()
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Length": "\(body.count)"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@Suite("HubDownloader", .serialized)
struct HubDownloaderTests {
    static func setUp(_ body: Data, status: Int = 200, sha: String? = nil, gated: Bool = false)
        -> (HubDownloader, HubFile, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("hub-\(UUID())")
        let file = HubFile(repoID: "org/repo", revision: "abc", path: "sub/w.safetensors", size: Int64(body.count),
                           sha256: sha ?? SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined(),
                           gated: gated)
        StubServer.lock.lock()
        StubServer.routes = ["/org/repo/resolve/abc/sub/w.safetensors": (status, body)]
        StubServer.hits = [:]
        StubServer.lock.unlock()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubServer.self]
        return (HubDownloader(root: root, token: nil, configuration: config), file, root)
    }

    static var hitCount: Int { StubServer.lock.lock(); defer { StubServer.lock.unlock() }; return StubServer.hits.values.reduce(0, +) }

    @Test func downloadsIntoRepoPathAndVerifies() async throws {
        let body = Data((0..<4096).map { UInt8($0 % 251) })
        let (d, file, root) = Self.setUp(body)
        defer { try? FileManager.default.removeItem(at: root) }
        try await d.download([file])
        let dest = root.appendingPathComponent("org/repo/sub/w.safetensors")
        #expect(try Data(contentsOf: dest) == body)
        #expect(!FileManager.default.fileExists(atPath: dest.path + ".part"))
        #expect(try HubDownloader.isComplete(file, at: dest))
    }

    @Test func presentAndVerifiedFileIsSkipped() async throws {
        let body = Data(repeating: 7, count: 100)
        let (d, file, root) = Self.setUp(body)
        defer { try? FileManager.default.removeItem(at: root) }
        try await d.download([file])
        try await d.download([file])
        #expect(Self.hitCount == 1)
    }

    @Test func wrongShaRemovesTheFileAndThrows() async throws {
        let body = Data(repeating: 1, count: 64)
        let (d, file, root) = Self.setUp(body, sha: String(repeating: "0", count: 64))
        defer { try? FileManager.default.removeItem(at: root) }
        await #expect(throws: SperidError.self) { try await d.download([file]) }
        #expect(!FileManager.default.fileExists(atPath: file.localURL(in: root).path))
    }

    @Test func forbiddenIsReportedAsGated() async throws {
        let (d, file, root) = Self.setUp(Data(), status: 403, gated: true)
        defer { try? FileManager.default.removeItem(at: root) }
        do {
            try await d.download([file])
            Issue.record("expected an error")
        } catch let SperidError.gatedRepository(message) {
            #expect(message.contains("https://huggingface.co/org/repo"))
        }
    }
}
