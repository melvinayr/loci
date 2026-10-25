// swift-tools-version: 6.0
import PackageDescription

// LociReader: a document reader built on Loci. The UI lives in `LociReaderKit` (macOS and iOS); the `LociReader`
// executable wraps it as a macOS app. For iOS, add `LociReaderKit` to an app target in Xcode and show `ReaderRootView`.
let package = Package(
    name: "LociReader",
    platforms: [.iOS("16.4"), .macOS("13.3")],
    products: [
        .library(name: "LociReaderKit", targets: ["LociReaderKit"]),
        .executable(name: "LociReader", targets: ["LociReader"]),
    ],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .target(
            name: "LociReaderKit",
            dependencies: [.product(name: "Loci", package: "loci"), .product(name: "LociHub", package: "loci")]
        ),
        .executableTarget(name: "LociReader", dependencies: ["LociReaderKit"]),
        .testTarget(
            name: "LociReaderKitTests",
            dependencies: ["LociReaderKit", .product(name: "Loci", package: "loci")]
        ),
    ]
)
