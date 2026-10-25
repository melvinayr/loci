/// A passage returned by `KnowledgeBase.search`.
public struct SearchResult: Sendable, Hashable {
    /// Scores from each retrieval stage; stages that did not run are nil.
    public struct Scores: Sendable, Hashable {
        /// BM25 relevance from full-text search; higher is better.
        public var bm25: Float?
        /// Cosine similarity from vector search.
        public var vector: Float?
        /// Reciprocal Rank Fusion score; set only in hybrid mode.
        public var fused: Float?
        /// Cross-encoder relevance logit; set only when the passage was reranked. Higher is better; the scale is model-specific.
        public var rerank: Float?
    }

    /// Identifies the passage.
    public let chunkID: ChunkID
    /// The document the passage comes from.
    public let document: Document
    /// 1-based, PDFs only.
    public let page: Int?
    /// Unicode-scalar offsets in the document's normalised text.
    public let range: Range<Int>
    /// The passage text, exactly the slice of the document text at `range`.
    public let text: String
    /// Scores of the best-scoring chunk in this passage.
    public let scores: Scores
}
