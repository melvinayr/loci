import Foundation

/// Locates GGUF files for integration tests. `LOCI_TEST_MODELS` overrides the default
/// `LLM/` directory at the repository root. Tests skip when a file is absent.
enum ModelFixtures {
    static let repositoryRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // LociIntegrationTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repository root

    static let directory: URL = {
        if let override = ProcessInfo.processInfo.environment["LOCI_TEST_MODELS"] {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        return repositoryRoot.appendingPathComponent("LLM", isDirectory: true)
    }()

    static let fixtureEmbedder = repositoryRoot.appendingPathComponent("Tests/Fixtures/models/bge-small-en-v1.5-q8_0.gguf")
    static let localEmbedder = directory.appendingPathComponent("embeddinggemma-300M-Q8_0.gguf")
    /// The local embedder if present, else the small fixture from `Scripts/fetch-test-models.sh`.
    static let embedder: URL = exists(localEmbedder) ? localEmbedder : fixtureEmbedder
    static let fixtureReranker = repositoryRoot.appendingPathComponent("Tests/Fixtures/models/qwen3-reranker-0.6b-q8_0.gguf")
    static let alternateReranker = repositoryRoot.appendingPathComponent("Tests/Fixtures/models/bge-reranker-v2-m3-Q4_K_M.gguf")
    static let fixtureGenerator = repositoryRoot.appendingPathComponent("Tests/Fixtures/models/qwen3-0.6b-q8_0.gguf")
    static let generator = directory.appendingPathComponent("gemma-4-E4B-it-Q4_0.gguf")

    static func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }
}
