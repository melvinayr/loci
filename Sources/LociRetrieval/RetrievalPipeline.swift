import LociStore

/// Hybrid retrieval over a store: BM25 and vector search, fused with RRF, then merged.
public struct RetrievalPipeline: Sendable {
    private let store: Store

    public init(store: Store) {
        self.store = store
    }

    public func retrieve(_ request: RetrievalRequest) async throws -> [Passage] {
        guard request.limit > 0 else { return [] }
        let pool = max(request.candidatePool, request.limit)
        let useKeyword = request.strategy != .vector
        let useVector = request.strategy != .keyword && request.queryVector != nil

        async let keywordHits = useKeyword
            ? store.keywordSearch(request.query, limit: pool, documentIDs: request.documentIDs) : []
        async let vectorHits = useVector
            ? store.vectorSearch(request.queryVector ?? [], limit: pool, documentIDs: request.documentIDs) : []
        let (keyword, vector) = try await (keywordHits, vectorHits)

        let bm25 = Dictionary(keyword.map { ($0.chunk.id, Float($0.score)) }, uniquingKeysWith: { first, _ in first })
        let cosine = Dictionary(vector.map { ($0.chunkID, $0.score) }, uniquingKeysWith: { first, _ in first })

        // Priority order of chunk IDs, with the fused score when two rankings were combined.
        var ordered: [(id: Int64, fused: Float?)]
        if useKeyword && useVector {
            ordered = RRF.fuse([keyword.map(\.chunk.id), vector.map(\.chunkID)]).map { ($0.id, Float($0.score)) }
        } else if useVector {
            ordered = vector.map { ($0.chunkID, nil) }
        } else {
            ordered = keyword.map { ($0.chunk.id, nil) }
        }

        var chunks = Dictionary(keyword.map { ($0.chunk.id, $0.chunk) }, uniquingKeysWith: { first, _ in first })
        let missing = ordered.map(\.id).filter { chunks[$0] == nil }
        for chunk in try await store.chunks(ids: missing) { chunks[chunk.id] = chunk }

        var candidates = ordered.compactMap { entry -> Candidate? in
            guard let chunk = chunks[entry.id] else { return nil }
            return Candidate(chunk: chunk, bm25: bm25[entry.id], vector: cosine[entry.id], fused: entry.fused, rerank: nil)
        }
        if let reranking = request.reranking, candidates.count > 1 {
            candidates = try await Self.rerank(candidates, query: request.query, reranking: reranking, limit: request.limit)
        }
        return try await PassageMerger.merge(candidates, limit: request.limit) { [store] in
            try await store.documentText(id: $0)
        }
    }

    /// Rescores the top `depth` candidates and orders them by that score. Candidates below the
    /// cut-off are dropped: the cross-encoder only ever sees, and so only ever promotes, the top few.
    private static func rerank(
        _ candidates: [Candidate], query: String, reranking: Reranking, limit: Int
    ) async throws -> [Candidate] {
        let top = Array(candidates.prefix(max(reranking.depth, limit)))
        let scores = try await reranking.score(query, top.map(\.chunk.text))
        guard scores.count == top.count else {
            throw StoreError.corrupted("the reranker returned \(scores.count) scores for \(top.count) passages")
        }
        var rescored = zip(top, scores).map { candidate, score -> Candidate in
            var updated = candidate
            updated.rerank = score.isFinite ? score : nil  // NaN or infinity cannot be ordered
            return updated
        }
        // Stable: equal scores keep their fused order.
        rescored = rescored.enumerated()
            .sorted { lhs, rhs in
                let (a, b) = (lhs.element.rerank ?? -.infinity, rhs.element.rerank ?? -.infinity)
                return a != b ? a > b : lhs.offset < rhs.offset
            }
            .map(\.element)
        return rescored
    }
}
