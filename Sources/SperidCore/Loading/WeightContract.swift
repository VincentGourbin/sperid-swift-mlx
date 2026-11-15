// WeightContract.swift - what a weight file must contain before a model loads it
// Copyright 2026 Vincent Gourbin

import Foundation

/// The loader's contract with one checkpoint file: exact tensor count, and the shapes of a few
/// anchor tensors that pin the architecture (a wrong file fails here, not as plausible garbage).
/// Counts and shapes come from the published headers (`Scripts/reference/dump_headers.py`).
public struct WeightContract: Sendable, Equatable {
    public let file: String
    public let tensorCount: Int
    /// Tensor name → exact shape.
    public let anchors: [String: [Int]]

    public init(file: String, tensorCount: Int, anchors: [String: [Int]]) {
        self.file = file
        self.tensorCount = tensorCount
        self.anchors = anchors
    }

    /// Throws `.weightMismatch` naming the first difference.
    public func verify(_ header: SafetensorsHeader) throws {
        guard header.entries.count == tensorCount else {
            throw SperidError.weightMismatch("\(file): \(header.entries.count) tensors, expected \(tensorCount)")
        }
        for (name, shape) in anchors.sorted(by: { $0.key < $1.key }) {
            guard let entry = header.entries[name] else {
                throw SperidError.weightMismatch("\(file): missing \(name)")
            }
            guard entry.shape == shape else {
                throw SperidError.weightMismatch("\(file): \(name) is \(entry.shape), expected \(shape)")
            }
        }
    }
}
