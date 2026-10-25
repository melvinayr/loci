import Foundation
import Testing
@testable import LociStore

private func unitVector(_ seed: Int, dimension: Int) -> [Float] {
    var generator = SeededGenerator(seed: UInt64(seed))
    var values = (0..<dimension).map { _ in Float.random(in: -1...1, using: &generator) }
    let norm = values.reduce(0) { $0 + $1 * $1 }.squareRoot()
    values = values.map { $0 / norm }
    return values
}

/// SplitMix64, so "random" test vectors are reproducible.
struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

private func dot(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }

@Suite("Float16 codec")
struct Float16CodecTests {
    @Test func roundTripsWithinHalfPrecision() {
        let values = unitVector(1, dimension: 384)
        let decoded = Float16Codec.decode(Float16Codec.encode(values))
        #expect(decoded.count == values.count)
        for (a, b) in zip(values, decoded) { #expect(abs(a - b) < 1e-3) }
    }

    @Test func handlesZerosSignsAndEmptyInput() {
        #expect(Float16Codec.decode(Float16Codec.encode([0, -0.5, 0.5, 1, -1])) == [0, -0.5, 0.5, 1, -1])
        #expect(Float16Codec.encode([]).isEmpty && Float16Codec.decode([]).isEmpty)
    }

    @Test func halvesTheStorage() {
        #expect(Float16Codec.encode([Float](repeating: 0.1, count: 100)).count * MemoryLayout<UInt16>.stride == 200)
    }
}

@Suite("Vector search")
struct VectorSearchTests {
    static let dimension = 32

    private func snapshot(count: Int, documents: Int = 3) -> (VectorSnapshot, [[Float]]) {
        var snapshot = VectorSnapshot()
        snapshot.dimension = Self.dimension
        snapshot.documentIDs = (0..<documents).map { "doc\($0)" }
        var vectors: [[Float]] = []
        for i in 0..<count {
            let vector = unitVector(i, dimension: Self.dimension)
            vectors.append(vector)
            snapshot.chunkIDs.append(Int64(i + 1))
            snapshot.documentOrdinals.append(Int32(i % documents))
            snapshot.components += Float16Codec.encode(vector)
        }
        return (snapshot, vectors)
    }

    @Test func matchesANaiveSearchAcrossBatchBoundaries() {
        // 10_000 rows spans three internal batches of 4096.
        let (snapshot, vectors) = snapshot(count: 10_000)
        let query = unitVector(77_777, dimension: Self.dimension)
        let hits = snapshot.search(query, limit: 10, allowedDocuments: nil)

        let expected = vectors.enumerated().map { (id: Int64($0.offset + 1), score: dot(query, $0.element)) }
            .sorted { $0.score > $1.score }.prefix(10)
        #expect(hits.count == 10)
        #expect(hits.map(\.score) == hits.map(\.score).sorted(by: >))
        // Half-precision storage can swap near-ties, so compare scores rather than IDs.
        for (hit, want) in zip(hits, expected) { #expect(abs(hit.score - want.score) < 5e-3) }
        #expect(Set(hits.map(\.chunkID)).intersection(expected.map(\.id)).count >= 8)
    }

    @Test func aStoredVectorIsItsOwnNearestNeighbour() {
        let (snapshot, vectors) = snapshot(count: 500)
        for index in [0, 123, 499] {
            let hit = snapshot.search(vectors[index], limit: 1, allowedDocuments: nil).first
            #expect(hit?.chunkID == Int64(index + 1))
            #expect((hit?.score ?? 0) > 0.99)
        }
    }

    @Test func documentFilterLimitAndShapeGuards() {
        let (snapshot, vectors) = snapshot(count: 90)
        let query = vectors[0]
        let filtered = snapshot.search(query, limit: 100, allowedDocuments: ["doc1"])
        #expect(filtered.count == 30)
        #expect(filtered.allSatisfy { ($0.chunkID - 1) % 3 == 1 })
        #expect(snapshot.search(query, limit: 5, allowedDocuments: []).isEmpty)
        #expect(snapshot.search(query, limit: 0, allowedDocuments: nil).isEmpty)
        #expect(snapshot.search([1, 2, 3], limit: 5, allowedDocuments: nil).isEmpty, "wrong dimension")
        #expect(VectorSnapshot().search(query, limit: 5, allowedDocuments: nil).isEmpty)
        #expect(snapshot.search(query, limit: 1000, allowedDocuments: nil).count == 90)
    }
}

private func makeStoreDocument(_ text: String) -> NewDocument {
    NewDocument(title: text, uri: nil, kind: "text", pageCount: nil, contentHash: "hash-\(text)", metadata: [:], text: text, pageRanges: nil)
}

private func makeStoreChunks(_ texts: [String]) -> [NewChunk] {
    var offset = 0
    return texts.enumerated().map { index, text in
        defer { offset += text.unicodeScalars.count + 2 }
        return NewChunk(ordinal: index, page: nil, start: offset, end: offset + text.unicodeScalars.count, headingPath: [], tokenCount: 2, text: text)
    }
}

@Suite("Vector storage")
struct VectorStorageTests {
    let embedder = StoreEmbedder(id: "m1", dimension: VectorSearchTests.dimension)

    private func withStore(_ body: (Store) async throws -> Void) async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try Store(directory: directory, fileProtection: .completeUntilFirstUserAuthentication)
        do { try await body(store) } catch { await store.close(); throw error }
        await store.close()
    }

    private func insert(_ store: Store, _ name: String, seeds: [Int]) async throws -> StoredDocument {
        let vectors = seeds.map { unitVector($0, dimension: embedder.dimension) }
        guard case .inserted(let stored) = try await store.insert(
            makeStoreDocument(name), chunks: makeStoreChunks(seeds.map { "\(name) chunk \($0)" }), vectors: vectors, embedder: embedder)
        else { throw StoreError.corrupted("expected insert") }
        return stored
    }

    @Test func searchSeesNewDocumentsImmediatelyAndForgetsRemovedOnes() async throws {
        try await withStore { store in
            let a = try await insert(store, "a", seeds: [1, 2, 3])
            let query = unitVector(2, dimension: embedder.dimension)
            let first = try await store.vectorSearch(query, limit: 1)
            #expect(first.count == 1 && first[0].score > 0.99)

            let b = try await insert(store, "b", seeds: [2])  // an identical vector in another document
            let both = try await store.vectorSearch(query, limit: 2)
            #expect(Set(both.map(\.chunkID)).count == 2 && both.allSatisfy { $0.score > 0.99 })

            #expect(try await store.remove(documentID: a.id))
            let afterRemoval = try await store.vectorSearch(query, limit: 10)
            let remaining = try await store.chunks(ids: afterRemoval.map(\.chunkID))
            #expect(remaining.allSatisfy { $0.documentID == b.id })
            #expect(try await store.stats().vectorCount == 1)
        }
    }

    @Test func documentFilterAndChunkLookupByID() async throws {
        try await withStore { store in
            let a = try await insert(store, "a", seeds: [1, 2])
            _ = try await insert(store, "b", seeds: [3])
            let hits = try await store.vectorSearch(unitVector(1, dimension: embedder.dimension), limit: 10, documentIDs: [a.id])
            let chunks = try await store.chunks(ids: hits.map(\.chunkID))
            #expect(chunks.count == 2 && chunks.allSatisfy { $0.documentID == a.id })
            #expect(chunks.map(\.id) == hits.map(\.chunkID), "lookup keeps the requested order")
            #expect(try await store.chunks(ids: [999_999]).isEmpty)
            #expect(try await store.chunks(ids: []).isEmpty)
            #expect(try await store.vectorSearch(unitVector(1, dimension: embedder.dimension), limit: 5, documentIDs: []).isEmpty)
        }
    }

    @Test func chunkLookupHandlesMoreIDsThanOneQueryHolds() async throws {
        try await withStore { store in
            let texts = (0..<1_200).map { "chunk \($0)" }
            _ = try await store.insert(makeStoreDocument("big"), chunks: makeStoreChunks(texts))
            let hits = try await store.keywordSearch("chunk", limit: 2_000)
            let found = try await store.chunks(ids: hits.map(\.chunk.id))
            #expect(found.count == 1_200)
        }
    }

    @Test func theFirstEmbedderIsRecordedAndAnotherIsRejected() async throws {
        try await withStore { store in
            #expect(try await store.storedEmbedder() == nil)
            _ = try await insert(store, "a", seeds: [1])
            #expect(try await store.storedEmbedder() == embedder)

            await #expect(throws: StoreError.embedderMismatch(stored: "m1", configured: "m2")) {
                try await store.insert(
                    makeStoreDocument("other"), chunks: makeStoreChunks(["x"]),
                    vectors: [unitVector(1, dimension: 8)], embedder: StoreEmbedder(id: "m2", dimension: 8))
            }
            #expect(try await store.documents().count == 1, "the rejected insert wrote nothing")
        }
    }

    @Test func mismatchedVectorShapesAreRejectedWithoutWriting() async throws {
        try await withStore { store in
            await #expect(throws: StoreError.self) {
                try await store.insert(makeStoreDocument("a"), chunks: makeStoreChunks(["x", "y"]), vectors: [unitVector(1, dimension: embedder.dimension)], embedder: embedder)
            }
            await #expect(throws: StoreError.self) {
                try await store.insert(makeStoreDocument("a"), chunks: makeStoreChunks(["x"]), vectors: [[1, 2]], embedder: embedder)
            }
            let stats = try await store.stats()
            #expect(stats.documentCount == 0)
        }
    }

    @Test func replaceVectorsSwapsEmbeddingsAndSetEmbedderSwitchesModel() async throws {
        try await withStore { store in
            let doc = try await insert(store, "a", seeds: [1, 2])
            let next = StoreEmbedder(id: "m2", dimension: 16)
            let fresh = [unitVector(10, dimension: 16), unitVector(11, dimension: 16)]
            try await store.replaceVectors(documentID: doc.id, vectors: fresh, embedder: next)
            try await store.setEmbedder(next)
            #expect(try await store.storedEmbedder() == next)

            let hit = try await store.vectorSearch(fresh[1], limit: 1).first
            let chunk = try await store.chunks(ids: hit.map { [$0.chunkID] } ?? []).first
            #expect(chunk?.ordinal == 1 && (hit?.score ?? 0) > 0.99)
            #expect(try await store.stats().vectorCount == 2)

            await #expect(throws: StoreError.self) {
                try await store.replaceVectors(documentID: doc.id, vectors: [fresh[0]], embedder: next)
            }
        }
    }

    @Test func rowsLeftOverFromAnInterruptedReindexAreIgnored() async throws {
        try await withStore { store in
            let doc = try await insert(store, "a", seeds: [1, 2])
            // Only the first chunk was re-embedded at the new dimension before the interruption.
            let chunks = try await store.chunks(documentID: doc.id)
            #expect(chunks.count == 2)
            let query16 = unitVector(5, dimension: 16)
            try await store.replaceVectors(documentID: doc.id, vectors: [query16, query16], embedder: StoreEmbedder(id: "m2", dimension: 16))
            let hits = try await store.vectorSearch(query16, limit: 10)
            #expect(hits.count == 2)
            // Searching with the old dimension finds nothing rather than misreading the new rows.
            #expect(try await store.vectorSearch(unitVector(5, dimension: embedder.dimension), limit: 10).isEmpty)
        }
    }

    @Test func concurrentWritesAndSearchesNeverSeeAStaleSnapshot() async throws {
        try await withStore { store in
            _ = try await insert(store, "seed", seeds: [0])
            let embedder = self.embedder
            try await withThrowingTaskGroup(of: Void.self) { group in
                for i in 1...12 {
                    group.addTask { _ = try await self.insert(store, "doc\(i)", seeds: [i]) }
                    group.addTask { _ = try await store.vectorSearch(unitVector(i, dimension: embedder.dimension), limit: 3) }
                }
                try await group.waitForAll()
            }
            // Once the writers are done, every stored vector is findable.
            for i in 0...12 {
                let hit = try await store.vectorSearch(unitVector(i, dimension: embedder.dimension), limit: 1).first
                #expect((hit?.score ?? 0) > 0.99, "vector \(i) missing from the snapshot")
            }
        }
    }

    @Test func vectorsAreRemovedWithTheirChunksInTheDatabase() async throws {
        try await withStore { store in
            let doc = try await insert(store, "a", seeds: [1, 2, 3])
            #expect(try await store.stats().vectorCount == 3)
            _ = try await store.remove(documentID: doc.id)
            #expect(try await store.stats().vectorCount == 0)
        }
    }
}
