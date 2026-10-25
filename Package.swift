// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Loci",
    platforms: [
        // Floors are set by the pinned llama.cpp binary (vtool minos: iOS 16.4, macOS 13.3).
        .iOS("16.4"),
        .macOS("13.3"),
    ],
    products: [
        .library(name: "Loci", targets: ["Loci"]),
        .library(name: "LociEval", targets: ["LociEval"]),
        .library(name: "LociHub", targets: ["LociHub"]),
        .executable(name: "loci-cli", targets: ["loci-cli"]),
    ],
    targets: [
        // Pinned llama.cpp release. Bump with Scripts/update-llama.sh.
        .binaryTarget(
            name: "llama",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b11849/llama-b11849-xcframework.zip",
            checksum: "d342c0462cb3e09582e428d61409d83e23f2a933ed6ab6a2644bc3d43f4901f9"
        ),

        // Layering (see docs/ARCHITECTURE.md): dependencies point down only.
        .target(name: "LociLlama", dependencies: ["llama"]),
        .target(name: "LociStore", linkerSettings: [.linkedLibrary("sqlite3")]),
        .target(name: "LociIngest"),
        .target(name: "LociRetrieval", dependencies: ["LociStore", "LociLlama"]),
        .target(name: "LociGrounding", dependencies: ["LociRetrieval", "LociLlama"]),
        .target(
            name: "Loci",
            dependencies: ["LociIngest", "LociStore", "LociRetrieval", "LociGrounding", "LociLlama"]
        ),
        .target(name: "LociEval", dependencies: ["Loci"]),
        .target(name: "LociHub", dependencies: ["Loci"]),

        .executableTarget(
            name: "loci-cli",
            dependencies: ["Loci", "LociEval", "LociHub"],
            path: "Examples/loci-cli"
        ),

        .testTarget(
            name: "LociTests",
            dependencies: ["Loci", "LociEval", "LociHub", "LociIngest", "LociStore", "LociRetrieval", "LociGrounding", "LociLlama"]
        ),
        // Measures real process memory, so it lives in its own target and runs on its own.
        .testTarget(name: "LociCalibrationTests", dependencies: ["LociLlama"]),
        .testTarget(
            name: "LociIntegrationTests",
            dependencies: ["Loci", "LociEval", "LociLlama", "llama"]
        ),
    ]
)
