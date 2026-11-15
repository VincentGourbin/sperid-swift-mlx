// SperidCore.swift - shared plumbing for the Speri Labs ports (Idol now, Iris later)
// Copyright 2026 Vincent Gourbin

/// What every Speri Labs model port shares and nothing model-specific: pinned Hub files,
/// the models directory, safetensors headers and weight contracts. Model code lives in its own
/// target (`IdolCore`), the way YuE2Swift keeps `SheetSage2Core` apart from `YuE2Core`.
public enum SperidCore {
    public static let version = "0.3.0"
}
