// swift-tools-version: 6.4
// Package.swift - Speri Labs models (Idol image-to-3D first) for Apple Silicon (MLX)
// Copyright 2026 Vincent Gourbin

import PackageDescription

let package = Package(
    name: "SperidSwift",
    platforms: [.macOS(.v15), .iOS(.v27)],
    products: [
        // MARK: - Libraries
        /// Shared by every Speri Labs port: weight files, safetensors contracts, downloads.
        .library(name: "SperidCore", targets: ["SperidCore"]),
        /// speridlabs/idol (and idol-turbo): one image → 3D Gaussian splat (.ply).
        .library(name: "IdolCore", targets: ["IdolCore"]),
        /// SwiftUI / RealityKit 27: the splat as an entity, a view, an offscreen renderer.
        .library(name: "IdolUI", targets: ["IdolUI"]),
        // MARK: - CLI Tools
        .executable(name: "idol", targets: ["IdolCLI"]),
        /// macOS 27 app: the speridlabs/idol Space locally (RealityKit Gaussian splats).
        .executable(name: "idol-studio", targets: ["IdolStudio"]),
    ],
    dependencies: [
        // A closed range of validated versions, not `exact`: host apps (Fluxforge Studio, with flux-2 and
        // yue2 at exact 0.31.6) must be able to resolve us next to their own pin. Both ends were checked on the
        // real weights — 0.31.6 (v0.3.1: 62 tests, parity CLI, end-to-end) and 0.32.3 (v0.3.0). Widen the upper
        // bound only after the same run: mlx-swift has broken API in minor releases before.
        .package(url: "https://github.com/ml-explore/mlx-swift", "0.31.6"..<"0.33.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
        // 1.5.1: compiles for iOS (xctrace code behind #if os(macOS)).
        .package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.5.1"),
    ],
    targets: [
        // MARK: - Libraries
        .target(
            name: "SperidCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "IdolCore",
            dependencies: [
                "SperidCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
            ]
        ),
        // MARK: - CLI Tools
        .executableTarget(
            name: "IdolCLI",
            dependencies: [
                "IdolCore",
                "SperidCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
            ]
        ),
        .target(
            name: "IdolUI",
            dependencies: ["IdolCore"]
        ),
        .executableTarget(
            name: "IdolStudio",
            dependencies: ["IdolCore", "IdolUI", "SperidCore"]
        ),
        // MARK: - Tests
        .testTarget(
            name: "SperidTests",
            dependencies: ["SperidCore", "IdolCore"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
