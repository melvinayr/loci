/// A retrieved passage: one chunk, or several overlapping or adjacent chunks merged.
/// `text` is exactly the slice of the document's normalised text at `range`.
public struct Passage: Sendable, Hashable {
    /// The best-scoring chunk among those merged into this passage.
    public let chunkID: Int64
    public let documentID: String
    public let page: Int?
    /// Unicode-scalar offsets in the document's normalised text.
    public let range: Range<Int>
    public let text: String
    public let headingPath: [String]
    public let bm25: Float?
    public let vector: Float?
    public let fused: Float?
    public let rerank: Float?
}

public enum RetrievalStrategy: Sendable, Hashable {
    case keyword, vector, hybrid
}

/// A cross-encoder pass over the best fused candidates.
public struct Reranking: Sendable {
    /// How many of the top fused candidates are rescored (never fewer than the request's `limit`).
    public var depth: Int
    /// One relevance score per passage text, in order; higher is better.
    public var score: @Sendable (_ query: String, _ passages: [String]) async throws -> [Float]

    public init(depth: Int, score: @escaping @Sendable (_ query: String, _ passages: [String]) async throws -> [Float]) {
        self.depth = depth
        self.score = score
    }
}

public struct RetrievalRequest: Sendable {
    public var query: String
    /// A unit vector from the store's embedder. Without one, hybrid falls back to keyword.
    public var queryVector: [Float]?
    public var limit: Int
    /// Candidates taken from each retriever before fusion.
    public var candidatePool: Int
    public var strategy: RetrievalStrategy
    public var documentIDs: Set<String>?
    /// Rescore the top candidates with a cross-encoder before merging; nil skips reranking.
    public var reranking: Reranking?

    public init(
        query: String, queryVector: [Float]? = nil, limit: Int = 5, candidatePool: Int = 40,
        strategy: RetrievalStrategy = .hybrid, documentIDs: Set<String>? = nil, reranking: Reranking? = nil
    ) {
        self.reranking = reranking
        self.query = query
        self.queryVector = queryVector
        self.limit = limit
        self.candidatePool = candidatePool
        self.strategy = strategy
        self.documentIDs = documentIDs
    }
}
