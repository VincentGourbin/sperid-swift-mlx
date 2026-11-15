// SafetensorsHeader.swift - read a safetensors header without loading any tensor
// Copyright 2026 Vincent Gourbin

import Foundation

/// The JSON header of a `.safetensors` file: 8-byte little-endian length, then
/// `{"name": {"dtype", "shape", "data_offsets"}, "__metadata__": {...}}`.
/// Reading it costs a few hundred kilobytes, never the weights: it is how a weight file is
/// checked against its contract before MLX maps it.
public struct SafetensorsHeader: Sendable, Equatable {
    public struct Entry: Sendable, Equatable {
        public let dtype: String
        public let shape: [Int]
    }

    public let entries: [String: Entry]
    public let metadata: [String: String]

    /// Headers larger than this are rejected as corrupt (the largest Idol header is ~70 KB).
    public static let maximumHeaderBytes = 64 << 20

    public init(entries: [String: Entry], metadata: [String: String] = [:]) {
        self.entries = entries
        self.metadata = metadata
    }

    public init(contentsOf url: URL) throws {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            throw SperidError.missingFile(url.lastPathComponent)
        }
        defer { try? handle.close() }
        guard let lengthBytes = try handle.read(upToCount: 8), lengthBytes.count == 8 else {
            throw SperidError.invalidSafetensors("\(url.lastPathComponent): shorter than 8 bytes")
        }
        let length = lengthBytes.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard length > 0, length <= UInt64(Self.maximumHeaderBytes) else {
            throw SperidError.invalidSafetensors("\(url.lastPathComponent): header length \(length)")
        }
        guard let json = try handle.read(upToCount: Int(length)), json.count == Int(length) else {
            throw SperidError.invalidSafetensors("\(url.lastPathComponent): truncated header")
        }
        try self.init(json: json, name: url.lastPathComponent)
    }

    public init(json: Data, name: String = "header") throws {
        guard let root = try? JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw SperidError.invalidSafetensors("\(name): header is not a JSON object")
        }
        var entries: [String: Entry] = [:]
        var metadata: [String: String] = [:]
        for (key, value) in root {
            if key == "__metadata__" {
                for (k, v) in (value as? [String: Any]) ?? [:] { metadata[k] = "\(v)" }
                continue
            }
            guard let object = value as? [String: Any], let dtype = object["dtype"] as? String,
                  let shape = object["shape"] as? [Int]
            else { throw SperidError.invalidSafetensors("\(name): bad entry \(key)") }
            entries[key] = Entry(dtype: dtype, shape: shape)
        }
        self.entries = entries
        self.metadata = metadata
    }

    /// Number of scalar parameters (all entries).
    public var parameterCount: Int {
        entries.values.reduce(0) { $0 + $1.shape.reduce(1, *) }
    }
}
