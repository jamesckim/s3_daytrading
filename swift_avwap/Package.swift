// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SwiftAVWAP",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "SwiftAVWAP",
            type: .dynamic,
            targets: ["SwiftAVWAP"]
        ),
    ],
    targets: [
        .target(
            name: "SwiftAVWAP",
            dependencies: [],
            swiftSettings: [
                .unsafeFlags(["-O", "-whole-module-optimization"])
            ]
        ),
    ]
)