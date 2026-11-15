// SperidError.swift - errors shared by every Speri Labs port
// Copyright 2026 Vincent Gourbin

import Foundation

public enum SperidError: Error, LocalizedError, Equatable {
    /// A weight file is missing from the local models directory.
    case missingFile(String)
    /// A safetensors file could not be parsed (truncated, bad header).
    case invalidSafetensors(String)
    /// A weight file does not match its contract (tensor count, names, shapes, dtypes) or its SHA-256.
    case weightMismatch(String)
    /// A request parameter is outside the range the reference implementation accepts.
    case invalidRequest(String)
    /// A gated Hugging Face repository refused the download (terms not accepted, or no token).
    case gatedRepository(String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let m): "missing file: \(m)"
        case .invalidSafetensors(let m): "invalid safetensors: \(m)"
        case .weightMismatch(let m): "weight mismatch: \(m)"
        case .invalidRequest(let m): "invalid request: \(m)"
        case .gatedRepository(let m): "gated repository: \(m)"
        }
    }
}
