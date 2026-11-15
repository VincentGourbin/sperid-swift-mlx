// WeightContractTests.swift - safetensors headers and loader contracts (no weights)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import IdolCore
@testable import SperidCore

@Suite("WeightContract")
struct WeightContractTests {
    static func writeSafetensors(_ json: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).safetensors")
        var length = UInt64(json.utf8.count).littleEndian
        var data = Data(bytes: &length, count: 8)
        data.append(Data(json.utf8))
        try data.write(to: url)
        return url
    }

    @Test func readsAHeaderWithoutTensors() throws {
        let url = try Self.writeSafetensors(
            #"{"__metadata__":{"format":"pt"},"a.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#)
        defer { try? FileManager.default.removeItem(at: url) }
        let h = try SafetensorsHeader(contentsOf: url)
        #expect(h.entries["a.weight"] == .init(dtype: "F32", shape: [2, 3]))
        #expect(h.metadata["format"] == "pt" && h.parameterCount == 6)
    }

    @Test func contractNamesTheFirstDifference() throws {
        let contract = WeightContract(file: "x", tensorCount: 1, anchors: ["a.weight": [2, 4]])
        let header = SafetensorsHeader(entries: ["a.weight": .init(dtype: "F32", shape: [2, 3])])
        #expect(throws: SperidError.weightMismatch("x: a.weight is [2, 3], expected [2, 4]")) {
            try contract.verify(header)
        }
    }

    @Test func pinnedFilesAreConsistent() {
        #expect(IdolWeights.idolFiles.reduce(0) { $0 + $1.size } == 3_078_981_231)
        for f in IdolWeights.idolFiles where f.path.hasSuffix(".safetensors") {
            #expect(f.sha256?.count == 64)
            #expect(f.url.absoluteString.contains(IdolWeights.idolRevision))
        }
    }
}
