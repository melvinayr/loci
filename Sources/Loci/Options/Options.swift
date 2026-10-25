import Foundation

/// How many models stay loaded at once, and so how much memory they hold.
public enum ModelResidency: Sendable, Hashable {
    /// One model resident at a time; loading another unloads the previous one. Slower, fits 6 GB phones. Default on iPhone.
    case lowMemory
    /// Embedder and reranker stay resident; the generator is loaded for one answer and released.
    case balanced
    /// Every model stays resident. For Macs.
    case keepResident
}

/// How many model layers run on the GPU.
public enum GPULayers: Sendable, Hashable {
    case all
    case none
    case count(Int)
}

/// Settings for opening a knowledge base.
public struct KnowledgeBaseOptions: Sendable {
    /// How many models stay loaded at once; see `ModelResidency`.
    public var residency: ModelResidency
    /// Data Protection class applied to the database, WAL and SHM files (iOS).
    public var fileProtection: FileProtectionType
    /// Layers offloaded to the GPU. Use `.none` for repeatable CPU-only runs.
    public var gpuLayers: GPULayers

    public init(
        residency: ModelResidency? = nil,
        fileProtection: FileProtectionType = .completeUntilFirstUserAuthentication,
        gpuLayers: GPULayers = .all
    ) {
        self.residency = residency ?? Self.defaultResidency
        self.fileProtection = fileProtection
        self.gpuLayers = gpuLayers
    }

    /// Defaults: low-memory residency on iOS, balanced elsewhere; all GPU layers; protection until first unlock.
    public static let `default` = KnowledgeBaseOptions()

    #if os(iOS)
    static let defaultResidency = ModelResidency.lowMemory
    #else
    static let defaultResidency = ModelResidency.balanced
    #endif
}

/// How a document is split into passages.
public enum ChunkerKind: String, Sendable, Codable, Hashable {
    case structural, sentence, tokenWindow
}

/// Settings for adding a document.
public struct IngestOptions: Sendable {
    /// How documents are split into passages.
    public var chunker: ChunkerKind
    /// Approximate passage size; tokens are estimated until the embedder tokenizer is wired in.
    public var targetChunkTokens: Int
    /// Approximate overlap between consecutive passages; clamped to at most half the target.
    public var overlapTokens: Int
    /// Stored per document and filterable with `MetadataFilter.metadata`.
    public var metadata: [String: String]

    public init(
        chunker: ChunkerKind = .structural,
        targetChunkTokens: Int = 256,
        overlapTokens: Int = 32,
        metadata: [String: String] = [:]
    ) {
        self.chunker = chunker
        self.targetChunkTokens = targetChunkTokens
        self.overlapTokens = overlapTokens
        self.metadata = metadata
    }

    public static let `default` = IngestOptions()
}

/// Which retrievers a search uses. `.hybrid` fuses both with Reciprocal Rank Fusion.
public enum RetrievalMode: Sendable, Hashable {
    case hybrid, vector, keyword
}

/// Restricts a search to some documents.
public enum MetadataFilter: Sendable, Hashable {
    case document([DocumentID])
    case metadata(String, String)
}

/// Settings for a search.
public struct RetrievalOptions: Sendable {
    /// Maximum results returned.
    public var limit: Int
    /// Candidates per retriever before fusion.
    public var candidatePool: Int
    /// Keyword, vector or hybrid retrieval.
    public var mode: RetrievalMode
    /// Rescore the best candidates with the reranker when one is configured; ignored otherwise.
    public var rerank: Bool
    /// Restrict results to some documents.
    public var filter: MetadataFilter?

    public init(
        limit: Int = 5,
        candidatePool: Int = 40,
        mode: RetrievalMode = .hybrid,
        rerank: Bool = true,
        filter: MetadataFilter? = nil
    ) {
        self.limit = limit
        self.candidatePool = candidatePool
        self.mode = mode
        self.rerank = rerank
        self.filter = filter
    }

    public static let `default` = RetrievalOptions()
}

/// Size and contents of a knowledge base.
public struct KnowledgeBaseStats: Sendable, Equatable {
    /// Documents stored.
    public let documentCount: Int
    /// Passages stored.
    public let chunkCount: Int
    /// Passages with an embedding; equals `chunkCount` unless a reindex is pending.
    public let vectorCount: Int
    /// Bytes used by the database, WAL and SHM files.
    public let databaseBytes: UInt64
    /// Identifies the embedding model the vectors were built with; nil before the first document.
    public let embedderID: String?
    /// Identifies the configured reranker (read from its file header); nil when none is configured or it cannot be found.
    public let rerankerID: String?
    /// Identifies the configured generator (read from its file header); nil when none is configured or it cannot be found.
    public let generatorID: String?
}
