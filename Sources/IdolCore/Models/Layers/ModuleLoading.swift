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
        try load(module, from: arrays, prefix: prefix, ignoring: ignored) { _ in dtype }
    }

    /// Same, with a per-key dtype (`nil` keeps the stored dtype). Each converted tensor is evaluated
    /// on its own (pitfall 8: no global eval of a half-converted model).
    public static func load(_ module: Module, from arrays: [String: MLXArray], prefix: String = "",
                            ignoring ignored: Set<String> = [], dtype: (String) -> DType?) throws {
        var selected: [String: MLXArray] = [:]
        for (key, value) in arrays where key.hasPrefix(prefix) {
            let name = String(key.dropFirst(prefix.count))
            if ignored.contains(name) { continue }
            if let target = dtype(name), target != value.dtype {
                let converted = value.asType(target)
                eval(converted)
                selected[name] = converted
            } else {
                selected[name] = value
            }
        }
        try module.update(parameters: ModuleParameters.unflattened(selected), verify: .all)
    }

    /// `convert_module_to_f16` of the reference: the Linear layers inside `blocks.*` in fp16; norms
    /// (LayerNorm32 weights, RMSNorm `gamma`) and everything outside the blocks stay fp32.
    public static func referencePrecision(blockDType: DType) -> (String) -> DType? {
        { name in
            guard name.hasPrefix("blocks.") else { return .float32 }
            return name.contains("norm") ? .float32 : blockDType
        }
    }
}
