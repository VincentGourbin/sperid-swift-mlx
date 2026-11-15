// ModuleLoading.swift - load upstream state_dict names into MLXNN modules, strictly
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

public enum ModuleLoading {
    /// Keys under `prefix` (prefix stripped), optionally cast to `dtype`, applied with
    /// `verify: .all` (every module parameter must be provided and no key may be left over —
    /// the equivalent of torch's `strict=True`).
    public static func load(_ module: Module, from arrays: [String: MLXArray], prefix: String = "",
                            dtype: DType? = nil, ignoring ignored: Set<String> = []) throws {
        var selected: [String: MLXArray] = [:]
        for (key, value) in arrays where key.hasPrefix(prefix) {
            let name = String(key.dropFirst(prefix.count))
            if ignored.contains(name) { continue }
            selected[name] = dtype.map { value.asType($0) } ?? value
        }
        try module.update(parameters: ModuleParameters.unflattened(selected), verify: .all)
    }
}
