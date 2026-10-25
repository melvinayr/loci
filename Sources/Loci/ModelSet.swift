import Foundation

/// Where a model file comes from.
public enum ModelSource: Sendable, Hashable {
    case file(URL)
    case bundle(name: String, bundle: Bundle = .main)
    /// Resolved by `LociHub`.
    case hub(repo: String, file: String, sha256: String?)
}

extension ModelSource {
    /// A short name for error messages: the file or resource name, never a full path.
    var displayName: String {
        switch self {
        case .file(let url): "“\(url.lastPathComponent)”"
        case .bundle(let name, _): "“\(name)”"
        case .hub(_, let file, _): "“\(file)” (download it with LociHub first)"
        }
    }
}

/// The models a knowledge base uses. Only the embedder is required.
public struct ModelSet: Sendable, Hashable {
    /// The embedding model; required, because it defines how passages are compared.
    public var embedder: ModelSource
    /// nil skips reranking.
    public var reranker: ModelSource?
    /// nil makes a retrieval-only knowledge base.
    public var generator: ModelSource?

    public init(embedder: ModelSource, reranker: ModelSource? = nil, generator: ModelSource? = nil) {
        self.embedder = embedder
        self.reranker = reranker
        self.generator = generator
    }

    private static let rerankerSource = ModelSource.hub(
        repo: "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF", file: "qwen3-reranker-0.6b-q8_0.gguf",
        sha256: "22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48")
    private static let generatorSource = ModelSource.hub(
        repo: "Qwen/Qwen3-4B-GGUF", file: "Qwen3-4B-Q4_K_M.gguf",
        sha256: "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5")

    /// English documents on a phone or Mac: bge-small (MIT), Qwen3-Reranker-0.6B and Qwen3-4B (Apache-2.0), about 3.2 GB on disk.
    /// Chosen by the measurements in `docs/MODELS.md`.
    public static let recommended = ModelSet(
        embedder: .hub(
            repo: "ggml-org/bge-small-en-v1.5-Q8_0-GGUF", file: "bge-small-en-v1.5-q8_0.gguf",
            sha256: "f046db1dc724cf4f6f0a0c5917e922823b73eb1d27b8f9a9c2797f7866974804"),
        reranker: rerankerSource, generator: generatorSource)
    /// The same reranker and generator with a larger embedder, nomic-embed-text-v1.5 (Apache-2.0), for Macs and Pro iPhones.
    public static let quality = ModelSet(
        embedder: .hub(
            repo: "nomic-ai/nomic-embed-text-v1.5-GGUF", file: "nomic-embed-text-v1.5.Q8_0.gguf",
            sha256: "3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7"),
        reranker: rerankerSource, generator: generatorSource)
}
