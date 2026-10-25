import Foundation
import os
import Testing
@testable import LociRetrieval
@testable import LociStore

@Suite("Reciprocal Rank Fusion")
struct RRFTests {
    @Test func scoresFollowTheFormula() {
        let fused = RRF.fuse([["a", "b", "c"], ["b", "c", "d"]])
        #expect(fused.map(\.id) == ["b", "c", "a", "d"])
        func score(_ id: String) -> Double { fused.first { $0.id == id }?.score ?? -1 }
        #expect(abs(score("a") - 1.0 / 61) < 1e-12)
        #expect(abs(score("b") - (1.0 / 62 + 1.0 / 61)) < 1e-12)
        #expect(abs(score("c") - (1.0 / 63 + 1.0 / 62)) < 1e-12)
        #expect(abs(score("d") - 1.0 / 63) < 1e-12)
    }

    @Test func kControlsHowMuchTopRanksDominate() {
        let rankings = [["x", "y"], ["z", "w", "y"]]
        // Large k flattens the curve, so appearing in both lists wins.
        #expect(RRF.fuse(rankings, k: 60).first?.id == "y")
        // k = 0 weights rank 1 as 1, rank 2 as 1/2, rank 3 as 1/3: the two rank-1 items beat y (1/2 + 1/3).
        #expect(RRF.fuse(rankings, k: 0).map(\.id) == ["x", "z", "y", "w"])
    }

    @Test func tiesKeepFirstSeenOrderAndDuplicatesCountOnce() {
        #expect(RRF.fuse([["a"], ["b"]]).map(\.id) == ["a", "b"])
        #expect(RRF.fuse([["b"], ["a"]]).map(\.id) == ["b", "a"])
        let duplicated = RRF.fuse([["a", "a", "a"]])
        #expect(duplicated.count == 1 && abs(duplicated[0].score - 1.0 / 61) < 1e-12)
    }

    @Test func emptyInputs() {
        #expect(RRF.fuse([[String]]()).isEmpty)
        #expect(RRF.fuse([[], []] as [[Int]]).isEmpty)
        #expect(RRF.fuse([["only"]]).map(\.id) == ["only"])
    }

    @Test func anItemInBothListsBeatsItemsInOne() {
        let fused = RRF.fuse([["k1", "both", "k2"], ["v1", "both", "v2"]])
        #expect(fused.first?.id == "both")
    }
}

// MARK: - Pipeline

private let dimension = 4

/// A document whose chunks are separated by `gaps[i]` newlines (default 2, a blank line). Chunks
/// two scalars apart count as adjacent and merge; use a larger gap to keep them separate.
private func makeDocument(
    _ store: Store, name: String, chunks texts: [String], vectors: [[Float]], pages: [Int?]? = nil,
    gap: Int = 2, gaps: [Int]? = nil
) async throws -> StoredDocument {
    let between = gaps ?? Array(repeating: gap, count: max(texts.count - 1, 0))
    var text = ""
    var chunks: [NewChunk] = []
    for (index, chunkText) in texts.enumerated() {
        let start = text.unicodeScalars.count
        text += chunkText
        chunks.append(NewChunk(
            ordinal: index, page: pages?[index], start: start, end: text.unicodeScalars.count,
            headingPath: ["H\(index)"], tokenCount: 3, text: chunkText))
        if index < between.count { text += String(repeating: "\n", count: between[index]) }
    }
    let document = NewDocument(
        title: name, uri: nil, kind: pages == nil ? "text" : "pdf", pageCount: nil, contentHash: "h-\(name)",
        metadata: [:], text: text, pageRanges: nil)
    guard case .inserted(let stored) = try await store.insert(
        document, chunks: chunks, vectors: vectors, embedder: StoreEmbedder(id: "m", dimension: dimension))
    else { throw StoreError.corrupted("expected insert") }
    return stored
}

private func withStore(_ body: (Store) async throws -> Void) async throws {
    let directory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = try Store(directory: directory, fileProtection: .completeUntilFirstUserAuthentication)
    do { try await body(store) } catch { await store.close(); throw error }
    await store.close()
}

private let e0: [Float] = [1, 0, 0, 0]
private func slice(_ text: String, _ range: Range<Int>) -> String {
    String(String.UnicodeScalarView(Array(text.unicodeScalars)[range]))
}

@Suite("Retrieval pipeline")
struct RetrievalPipelineTests {
    @Test func hybridCombinesKeywordAndVectorEvidence() async throws {
        try await withStore { store in
            _ = try await makeDocument(
                store, name: "d",
                chunks: ["alpha only keyword match", "completely different words here", "zeta vector favourite text", "unrelated filler content"],
                vectors: [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], gap: 10)
            let pipeline = RetrievalPipeline(store: store)

            let keyword = try await pipeline.retrieve(RetrievalRequest(query: "alpha", queryVector: e0, strategy: .keyword))
            #expect(keyword.map(\.text) == ["alpha only keyword match"])
            #expect(keyword[0].bm25 != nil && keyword[0].vector == nil && keyword[0].fused == nil)

            let vector = try await pipeline.retrieve(RetrievalRequest(query: "alpha", queryVector: e0, limit: 1, strategy: .vector))
            #expect(vector.first?.text == "zeta vector favourite text")
            #expect(vector[0].bm25 == nil && (vector[0].vector ?? 0) > 0.99 && vector[0].fused == nil)

            let hybrid = try await pipeline.retrieve(RetrievalRequest(query: "alpha", queryVector: e0, limit: 4, strategy: .hybrid))
            #expect(hybrid.map(\.text).contains("alpha only keyword match"))
            #expect(hybrid.map(\.text).contains("zeta vector favourite text"))
            #expect(hybrid.allSatisfy { $0.fused != nil })
        }
    }

    @Test func hybridWithoutAQueryVectorIsKeywordOnly() async throws {
        try await withStore { store in
            _ = try await makeDocument(store, name: "d", chunks: ["alpha words", "beta words"], vectors: [e0, [0, 1, 0, 0]])
            let results = try await RetrievalPipeline(store: store).retrieve(RetrievalRequest(query: "alpha", strategy: .hybrid))
            #expect(results.map(\.text) == ["alpha words"])
            #expect(results[0].fused == nil)
        }
    }

    @Test func hybridOrderIsExactlyRRFOfTheTwoRetrieverOrders() async throws {
        try await withStore { store in
            _ = try await makeDocument(
                store, name: "d",
                chunks: ["alpha beta gamma", "alpha beta", "alpha", "delta epsilon", "zeta eta"],
                vectors: [[0, 1, 0, 0], [0.6, 0.8, 0, 0], [0.2, 0.9, 0.3, 0], [1, 0, 0, 0], [0.9, 0, 0.4, 0]], gap: 10)
            let pipeline = RetrievalPipeline(store: store)
            func order(_ strategy: RetrievalStrategy) async throws -> [String] {
                try await pipeline.retrieve(RetrievalRequest(query: "alpha beta gamma", queryVector: e0, limit: 5, strategy: strategy)).map(\.text)
            }
            let keyword = try await order(.keyword)
            let vector = try await order(.vector)
            #expect(keyword.count == 3 && vector.count == 5)
            let expected = RRF.fuse([keyword, vector]).map(\.id)
            #expect(try await order(.hybrid) == expected)
            #expect(expected != keyword && expected != vector, "fusion should change the order here")
        }
    }

    @Test func documentFilterAppliesToBothRetrievers() async throws {
        try await withStore { store in
            let a = try await makeDocument(store, name: "a", chunks: ["shared term a"], vectors: [e0])
            _ = try await makeDocument(store, name: "b", chunks: ["shared term b"], vectors: [e0])
            let results = try await RetrievalPipeline(store: store).retrieve(
                RetrievalRequest(query: "shared", queryVector: e0, strategy: .hybrid, documentIDs: [a.id]))
            #expect(results.map(\.documentID) == [a.id])
        }
    }

    @Test func limitAndPoolAreRespected() async throws {
        try await withStore { store in
            let texts = (0..<12).map { "common word number \($0 * 100)" }
            // Interleave unrelated chunks so the matches do not touch and merge.
            var chunks: [String] = [], vectors: [[Float]] = []
            for text in texts { chunks += [text, "separator filler \(text.count)"]; vectors += [e0, [0, 1, 0, 0]] }
            _ = try await makeDocument(store, name: "d", chunks: chunks, vectors: vectors, gap: 10)
            let pipeline = RetrievalPipeline(store: store)
            let results = try await pipeline.retrieve(RetrievalRequest(query: "common", queryVector: e0, limit: 3, candidatePool: 2, strategy: .hybrid))
            #expect(results.count == 3, "a pool smaller than the limit is raised to the limit")
            #expect(try await pipeline.retrieve(RetrievalRequest(query: "common", limit: 0)).isEmpty)
        }
    }
}

@Suite("Passage merging")
struct PassageMergingTests {
    private func vectors(_ cosines: [Float]) -> [[Float]] {
        cosines.map { [$0, (1 - $0 * $0).squareRoot(), 0, 0] }
    }

    @Test func adjacentChunksMergeIntoOnePassageWithExactText() async throws {
        try await withStore { store in
            let doc = try await makeDocument(
                store, name: "d", chunks: ["first chunk", "second chunk", "third chunk", "far away chunk"],
                vectors: vectors([0.9, 0.8, 0.1, 0.05]), gaps: [2, 2, 10])
            let results = try await RetrievalPipeline(store: store).retrieve(
                RetrievalRequest(query: "x", queryVector: e0, limit: 2, strategy: .vector))
            let text = try #require(try await store.documentText(id: doc.id))

            #expect(results.count == 2)
            #expect(results.last?.text == "far away chunk")
            let merged = try #require(results.first)
            #expect(merged.text == "first chunk\n\nsecond chunk\n\nthird chunk")
            #expect(merged.text == slice(text, merged.range))
            #expect(merged.headingPath == ["H0"], "heading path comes from the earliest chunk")
            #expect(merged.vector.map { $0 > 0.89 } == true, "scores come from the best chunk")
            let best = try #require(try await store.chunks(ids: [merged.chunkID]).first)
            #expect(best.text == "first chunk")
        }
    }

    @Test func aChunkBridgingTwoGroupsJoinsThemAll() async throws {
        try await withStore { store in
            // c0 and c2 are retrieved first and stay apart; c1 arrives last and touches both.
            let doc = try await makeDocument(
                store, name: "d", chunks: ["left side", "bridge", "right side", "another far"],
                vectors: vectors([1.0, 0.5, 0.9, 0.0]), gaps: [2, 2, 10])
            let results = try await RetrievalPipeline(store: store).retrieve(
                RetrievalRequest(query: "x", queryVector: e0, limit: 3, strategy: .vector))
            let text = try #require(try await store.documentText(id: doc.id))
            #expect(results.map(\.text) == ["left side\n\nbridge\n\nright side", "another far"])
            for passage in results { #expect(passage.text == slice(text, passage.range)) }
        }
    }

    @Test func overlappingChunksMerge() async throws {
        try await withStore { store in
            let text = "abcdefghij klmnopqrst uvwxyz"
            let document = NewDocument(
                title: "o", uri: nil, kind: "text", pageCount: nil, contentHash: "h-o", metadata: [:], text: text, pageRanges: nil)
            let chunks = [
                NewChunk(ordinal: 0, page: nil, start: 0, end: 16, headingPath: [], tokenCount: 3, text: slice(text, 0..<16)),
                NewChunk(ordinal: 1, page: nil, start: 11, end: 28, headingPath: [], tokenCount: 3, text: slice(text, 11..<28)),
            ]
            _ = try await store.insert(
                document, chunks: chunks, vectors: vectors([0.9, 0.8]), embedder: StoreEmbedder(id: "m", dimension: dimension))
            let results = try await RetrievalPipeline(store: store).retrieve(
                RetrievalRequest(query: "x", queryVector: e0, limit: 5, strategy: .vector))
            #expect(results.count == 1)
            #expect(results[0].range == 0..<28 && results[0].text == text)
        }
    }

    @Test func chunksOnDifferentPagesNeverMerge() async throws {
        try await withStore { store in
            _ = try await makeDocument(
                store, name: "pdf", chunks: ["page one tail", "page two head"], vectors: vectors([0.9, 0.8]), pages: [1, 2])
            let results = try await RetrievalPipeline(store: store).retrieve(
                RetrievalRequest(query: "x", queryVector: e0, limit: 5, strategy: .vector))
            #expect(results.count == 2)
            #expect(Set(results.compactMap(\.page)) == [1, 2])
        }
    }

    @Test func chunksOfDifferentDocumentsNeverMerge() async throws {
        try await withStore { store in
            _ = try await makeDocument(store, name: "a", chunks: ["same place"], vectors: vectors([0.9]))
            _ = try await makeDocument(store, name: "b", chunks: ["same place"], vectors: vectors([0.8]))
            let results = try await RetrievalPipeline(store: store).retrieve(
                RetrievalRequest(query: "x", queryVector: e0, limit: 5, strategy: .vector))
            #expect(results.count == 2)
        }
    }

    @Test func mergingNeverExceedsTheLimit() async throws {
        try await withStore { store in
            let chunks = (0..<10).map { "chunk number \($0)" }
            _ = try await makeDocument(store, name: "d", chunks: chunks, vectors: vectors((0..<10).map { 0.9 - Float($0) * 0.05 }))
            for limit in 1...4 {
                let results = try await RetrievalPipeline(store: store).retrieve(
                    RetrievalRequest(query: "x", queryVector: e0, limit: limit, strategy: .vector))
                #expect(results.count <= limit)
                #expect(!results.isEmpty)
            }
        }
    }
}


@Suite("Reranking stage")
struct RerankingStageTests {
    private func scorer(_ log: RerankLog? = nil, _ rule: @escaping @Sendable (String) -> Float) -> Reranking {
        Reranking(depth: 20) { query, passages in
            log?.record(query: query, passages: passages)
            return passages.map(rule)
        }
    }

    private func seed(_ store: Store, _ texts: [String]) async throws {
        _ = try await makeDocument(store, name: "d", chunks: texts, vectors: texts.indices.map { _ in e0 }, gap: 10)
    }

    @Test func theCrossEncoderOrderReplacesTheFusedOrderAndScoresAreReported() async throws {
        try await withStore { store in
            try await seed(store, ["alpha one", "alpha two needle", "alpha three", "alpha four"])
            let pipeline = RetrievalPipeline(store: store)
            let plain = try await pipeline.retrieve(RetrievalRequest(query: "alpha", strategy: .keyword))
            let reranked = try await pipeline.retrieve(RetrievalRequest(
                query: "alpha", strategy: .keyword, reranking: scorer { $0.contains("needle") ? 5 : 1 }))

            #expect(plain.first?.text != "alpha two needle", "the needle is not first without reranking")
            #expect(reranked.first?.text == "alpha two needle")
            #expect(reranked.first?.rerank == 5 && reranked.dropFirst().allSatisfy { $0.rerank == 1 })
            #expect(reranked.allSatisfy { $0.bm25 != nil }, "earlier scores are kept")
            #expect(plain.allSatisfy { $0.rerank == nil })
        }
    }

    @Test func equalScoresKeepTheFusedOrder() async throws {
        try await withStore { store in
            try await seed(store, ["alpha one", "alpha two", "alpha three"])
            let pipeline = RetrievalPipeline(store: store)
            let plain = try await pipeline.retrieve(RetrievalRequest(query: "alpha", strategy: .keyword)).map(\.text)
            let flat = try await pipeline.retrieve(RetrievalRequest(query: "alpha", strategy: .keyword, reranking: scorer { _ in 1 })).map(\.text)
            #expect(flat == plain)
        }
    }

    @Test func onlyTheTopDepthCandidatesAreScoredAndDepthIsRaisedToTheLimit() async throws {
        try await withStore { store in
            try await seed(store, (0..<10).map { "alpha item \($0)" })
            let pipeline = RetrievalPipeline(store: store)

            let shallow = RerankLog()
            var request = RetrievalRequest(query: "alpha", limit: 2, strategy: .keyword)
            request.reranking = Reranking(depth: 3) { q, p in shallow.record(query: q, passages: p); return p.map { _ in 0 } }
            let results = try await pipeline.retrieve(request)
            #expect(shallow.passages.map(\.count) == [3])
            #expect(results.count == 2)

            let raised = RerankLog()
            request.limit = 4
            request.reranking = Reranking(depth: 1) { q, p in raised.record(query: q, passages: p); return p.map { _ in 0 } }
            #expect(try await pipeline.retrieve(request).count == 4)
            #expect(raised.passages.map(\.count) == [4], "depth never drops below the limit")
        }
    }

    @Test func theReRankerSeesTheQueryAndEachChunksText() async throws {
        try await withStore { store in
            try await seed(store, ["alpha one", "alpha two"])
            let log = RerankLog()
            _ = try await RetrievalPipeline(store: store).retrieve(RetrievalRequest(
                query: "the actual question", strategy: .vector, reranking: scorer(log) { _ in 0 }))
            #expect(log.queries.isEmpty, "vector mode without a query vector retrieves nothing, so nothing is reranked")
            _ = try await RetrievalPipeline(store: store).retrieve(RetrievalRequest(
                query: "alpha question", strategy: .keyword, reranking: scorer(log) { _ in 0 }))
            #expect(log.queries == ["alpha question"])
            #expect(Set(log.passages.flatMap { $0 }) == ["alpha one", "alpha two"])
        }
    }

    @Test func aSingleCandidateIsNotWorthScoring() async throws {
        try await withStore { store in
            try await seed(store, ["only alpha chunk"])
            let log = RerankLog()
            let results = try await RetrievalPipeline(store: store).retrieve(RetrievalRequest(
                query: "alpha", strategy: .keyword, reranking: scorer(log) { _ in 9 }))
            #expect(results.count == 1 && log.queries.isEmpty && results[0].rerank == nil)
        }
    }

    @Test func aWrongNumberOfScoresIsAnErrorNotASilentMisordering() async throws {
        try await withStore { store in
            try await seed(store, ["alpha one", "alpha two", "alpha three"])
            let broken = Reranking(depth: 20) { _, passages in Array(repeating: 1, count: passages.count - 1) }
            await #expect(throws: StoreError.self) {
                try await RetrievalPipeline(store: store).retrieve(RetrievalRequest(query: "alpha", strategy: .keyword, reranking: broken))
            }
        }
    }

    @Test func nonFiniteScoresAreSortedLastAndNotReported() async throws {
        try await withStore { store in
            try await seed(store, ["alpha one", "alpha two", "alpha three"])
            let scores: [String: Float] = ["alpha one": .nan, "alpha two": 2, "alpha three": -.infinity]
            let results = try await RetrievalPipeline(store: store).retrieve(RetrievalRequest(
                query: "alpha", strategy: .keyword, reranking: scorer { scores[$0] ?? 0 }))
            #expect(results.first?.text == "alpha two" && results.first?.rerank == 2)
            #expect(results.count == 3 && results.dropFirst().allSatisfy { $0.rerank == nil || $0.rerank?.isFinite == true })
        }
    }

    @Test func errorsFromTheRerankerPropagate() async throws {
        struct Boom: Error {}
        try await withStore { store in
            try await seed(store, ["alpha one", "alpha two"])
            let failing = Reranking(depth: 20) { _, _ in throw Boom() }
            await #expect(throws: Boom.self) {
                try await RetrievalPipeline(store: store).retrieve(RetrievalRequest(query: "alpha", strategy: .keyword, reranking: failing))
            }
        }
    }

    @Test func aMergedPassageCarriesItsBestChunksRerankScore() async throws {
        try await withStore { store in
            // Adjacent chunks (gap 2) merge into one passage.
            _ = try await makeDocument(
                store, name: "d", chunks: ["alpha first", "alpha second best", "alpha third"],
                vectors: [e0, e0, e0], gap: 2)
            let results = try await RetrievalPipeline(store: store).retrieve(RetrievalRequest(
                query: "alpha", strategy: .keyword, reranking: scorer { $0.contains("best") ? 7 : 1 }))
            #expect(results.count == 1)
            #expect(results[0].rerank == 7)
            #expect(results[0].text == "alpha first\n\nalpha second best\n\nalpha third")
            let best = try await store.chunks(ids: [results[0].chunkID]).first
            #expect(best?.text == "alpha second best")
        }
    }
}

final class RerankLog: Sendable {
    private let state = OSAllocatedUnfairLock(initialState: (queries: [String](), passages: [[String]]()))
    func record(query: String, passages: [String]) { state.withLock { $0.queries.append(query); $0.passages.append(passages) } }
    var queries: [String] { state.withLock { $0.queries } }
    var passages: [[String]] { state.withLock { $0.passages } }
}
