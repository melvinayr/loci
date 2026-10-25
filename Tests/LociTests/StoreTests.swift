import Foundation
import SQLite3
import Testing
@testable import LociStore

private func makeDocument(
    title: String = "Doc", uri: String? = nil, text: String = "alpha beta gamma", hash: String? = nil,
    metadata: [String: String] = [:], pageRanges: [Range<Int>?]? = nil
) -> NewDocument {
    NewDocument(
        title: title, uri: uri, kind: "text", pageCount: pageRanges?.count, contentHash: hash ?? "hash-\(text)",
        metadata: metadata, text: text, pageRanges: pageRanges)
}

private func makeChunks(_ texts: [String]) -> [NewChunk] {
    var offset = 0
    return texts.enumerated().map { index, text in
        defer { offset += text.unicodeScalars.count + 1 }
        return NewChunk(
            ordinal: index, page: nil, start: offset, end: offset + text.unicodeScalars.count,
            headingPath: ["H"], tokenCount: 3, text: text)
    }
}

private func withStore(_ body: (Store, URL) async throws -> Void) async throws {
    let directory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = try Store(directory: directory, fileProtection: .completeUntilFirstUserAuthentication)
    do {
        try await body(store, directory)
    } catch {
        await store.close()
        throw error
    }
    await store.close()
}

@Suite("Store")
struct StoreTests {
    @Test func insertsAndReadsBack() async throws {
        try await withStore { store, _ in
            let outcome = try await store.insert(
                makeDocument(title: "T", uri: "u://1", metadata: ["k": "v"], pageRanges: [0..<5, nil]),
                chunks: makeChunks(["alpha", "beta"]))
            guard case .inserted(let stored) = outcome else { Issue.record("expected insert"); return }
            #expect(stored.chunkCount == 2)

            let documents = try await store.documents()
            #expect(documents == [stored])
            #expect(documents[0].metadata == ["k": "v"])
            #expect(try await store.documentText(id: stored.id) == "alpha beta gamma")
            #expect(try await store.pageRanges(documentID: stored.id) == [0..<5, nil])

            let chunks = try await store.chunks(documentID: stored.id)
            #expect(chunks.map(\.text) == ["alpha", "beta"])
            #expect(chunks.map(\.ordinal) == [0, 1])
            #expect(chunks[0].headingPath == ["H"])
        }
    }

    @Test func reAddingIdenticalContentWritesNothing() async throws {
        try await withStore { store, _ in
            let first = try await store.insert(makeDocument(), chunks: makeChunks(["alpha"]))
            let second = try await store.insert(makeDocument(title: "Other title"), chunks: makeChunks(["alpha", "again"]))
            guard case .inserted(let a) = first, case .existing(let b) = second else {
                Issue.record("expected insert then existing"); return
            }
            #expect(a.id == b.id)
            let stats = try await store.stats()
            #expect(stats.documentCount == 1 && stats.chunkCount == 1)
        }
    }

    @Test func changedContentAtSameURIReplacesTheOldVersion() async throws {
        try await withStore { store, _ in
            _ = try await store.insert(makeDocument(uri: "file:///a", text: "old words"), chunks: makeChunks(["old words"]))
            _ = try await store.insert(makeDocument(uri: "file:///a", text: "new words"), chunks: makeChunks(["new words"]))
            #expect(try await store.documents().count == 1)
            #expect(try await store.keywordSearch("old", limit: 5).isEmpty)
            #expect(try await store.keywordSearch("new", limit: 5).count == 1)
        }
    }

    @Test func removeCascadesToChunksAndFullTextIndex() async throws {
        try await withStore { store, _ in
            guard case .inserted(let stored) = try await store.insert(makeDocument(), chunks: makeChunks(["alpha", "beta"])) else { return }
            #expect(try await store.keywordSearch("alpha", limit: 5).count == 1)
            #expect(try await store.remove(documentID: stored.id))
            #expect(try await store.remove(documentID: stored.id) == false)
            let stats = try await store.stats()
            #expect(stats.documentCount == 0 && stats.chunkCount == 0)
            #expect(try await store.keywordSearch("alpha", limit: 5).isEmpty)
        }
    }

    @Test func aTransactionThatThrowsLeavesNoRows() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let connection = try SQLiteConnection(path: directory.appendingPathComponent("x.sqlite").path, readOnly: false)
        try Schema.migrate(connection)

        struct Boom: Error {}
        #expect(throws: Boom.self) {
            try connection.transaction {
                try connection.execute("INSERT INTO documents VALUES ('d','T',NULL,'text',NULL,'h','{}',0,'x',NULL)")
                try connection.execute(
                    "INSERT INTO chunks(document_id,ordinal,start,\"end\",heading_path,token_count,text) VALUES ('d',0,0,1,'[]',1,'x')")
                throw Boom()
            }
        }
        let counts = try connection.counts()
        #expect(counts.documents == 0 && counts.chunks == 0)
        #expect(try connection.keywordSearch(match: "\"x\"", limit: 5, documentIDs: nil).isEmpty)
    }

    @Test func keywordSearchRanksByBM25AndStems() async throws {
        try await withStore { store, _ in
            _ = try await store.insert(
                makeDocument(text: "doc"),
                chunks: makeChunks([
                    "participants were randomised into two groups",
                    "the weather was mild all week",
                    "randomised randomised randomised trial of participants",
                ]))
            let hits = try await store.keywordSearch("randomised participants", limit: 5)
            #expect(hits.count == 2)
            #expect(hits[0].score >= hits[1].score)
            #expect(hits.allSatisfy { $0.score > 0 })
            // Porter stemming: "participant" matches "participants".
            #expect(try await store.keywordSearch("participant", limit: 5).count == 2)
        }
    }

    @Test func keywordSearchIsSafeAgainstFTSSyntax() async throws {
        try await withStore { store, _ in
            _ = try await store.insert(makeDocument(), chunks: makeChunks(["alpha beta"]))
            for query in ["\"", "alpha AND", "NEAR(", "alpha*", "col:alpha", "-alpha", "'; DROP TABLE chunks;--", "", "   ", "!!!"] {
                _ = try await store.keywordSearch(query, limit: 5)
            }
            #expect(try await store.stats().chunkCount == 1)
        }
    }

    @Test func documentFilterAppliesInlineAndAbovePostFilterThreshold() async throws {
        try await withStore { store, _ in
            guard case .inserted(let a) = try await store.insert(makeDocument(text: "a"), chunks: makeChunks(["shared word a"])),
                  case .inserted(let b) = try await store.insert(makeDocument(text: "b"), chunks: makeChunks(["shared word b"])) else { return }

            let inline = try await store.keywordSearch("shared", limit: 5, documentIDs: [a.id])
            #expect(inline.map(\.chunk.documentID) == [a.id])
            #expect(try await store.keywordSearch("shared", limit: 5, documentIDs: []).isEmpty)

            // Force the post-filter path with a tiny inline limit.
            let post = try await store.keywordSearch("shared", limit: 5, documentIDs: [a.id, b.id], inlineFilterLimit: 1)
            #expect(Set(post.map(\.chunk.documentID)) == [a.id, b.id])
            let postOne = try await store.keywordSearch("shared", limit: 1, documentIDs: [a.id, b.id], inlineFilterLimit: 1)
            #expect(postOne.count == 1)
        }
    }

    @Test func metaRoundTrips() async throws {
        try await withStore { store, _ in
            #expect(try await store.meta("schema_version") == String(Schema.currentVersion))
            try await store.setMeta("embedder_id", "x")
            try await store.setMeta("embedder_id", "y")
            #expect(try await store.meta("embedder_id") == "y")
        }
    }

    @Test func reopeningKeepsDataAndConcurrentReadsDuringWritesWork() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try Store(directory: directory, fileProtection: .completeUntilFirstUserAuthentication)
        _ = try await store.insert(makeDocument(), chunks: makeChunks(["alpha"]))

        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask { _ = try await store.insert(makeDocument(text: "doc \(i)"), chunks: makeChunks(["word\(i)"])) }
                group.addTask { _ = try await store.keywordSearch("alpha", limit: 3) }
            }
            try await group.waitForAll()
        }
        #expect(try await store.stats().documentCount == 11)
        await store.close()

        let reopened = try Store(directory: directory, fileProtection: .completeUntilFirstUserAuthentication)
        #expect(try await reopened.stats().documentCount == 11)
        await reopened.close()
    }

    @Test func usingAClosedStoreThrowsInsteadOfCrashing() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try Store(directory: directory, fileProtection: .completeUntilFirstUserAuthentication)
        await store.close()
        await #expect(throws: StoreError.self) { try await store.documents() }
        await #expect(throws: StoreError.self) { try await store.insert(makeDocument(), chunks: []) }
    }

    @Test func managedFilesIncludeWALAndSHM() {
        #expect(Store.managedFileNames() == ["loci.sqlite", "loci.sqlite-wal", "loci.sqlite-shm"])
    }
}

@Suite("Store migrations")
struct MigrationTests {
    @Test func emptyDatabaseMigratesToCurrentVersion() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let connection = try SQLiteConnection(path: directory.appendingPathComponent("x.sqlite").path, readOnly: false)
        #expect(try Schema.storedVersion(connection) == 0)
        try Schema.migrate(connection)
        #expect(try Schema.storedVersion(connection) == Schema.currentVersion)
        try Schema.migrate(connection)  // idempotent
        #expect(try Schema.storedVersion(connection) == Schema.currentVersion)
    }

    @Test func databaseFromANewerVersionIsRejected() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let connection = try SQLiteConnection(path: directory.appendingPathComponent("x.sqlite").path, readOnly: false)
        try Schema.migrate(connection)
        try connection.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
        #expect(throws: StoreError.corrupted("the store was written by a newer version of Loci")) {
            try Schema.migrate(connection)
        }
    }

    @Test func version1DatabaseMigratesToV2KeepingItsData() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent(Store.databaseName).path
        do {
            let connection = try SQLiteConnection(path: path, readOnly: false)
            try Schema.migrate(connection, to: 1)
            #expect(try Schema.storedVersion(connection) == 1)
            try connection.execute(
                "INSERT INTO documents VALUES ('d1','Old','u','text',NULL,'h','{}',0,'legacy words',NULL)", cached: false)
            try connection.execute(
                "INSERT INTO chunks(document_id,ordinal,page,start,\"end\",heading_path,token_count,text) VALUES ('d1',0,NULL,0,12,'[]',3,'legacy words')",
                cached: false)
            // v1 has no vectors table.
            #expect(throws: StoreError.self) { try connection.counts() }
            connection.close()
        }
        let store = try Store(directory: directory, fileProtection: .completeUntilFirstUserAuthentication)
        #expect(try await store.meta("schema_version") == "2")
        #expect(try await store.documents().map(\.title) == ["Old"])
        #expect(try await store.keywordSearch("legacy", limit: 5).count == 1)
        let stats = try await store.stats()
        #expect(stats.chunkCount == 1 && stats.vectorCount == 0)
        #expect(try await store.storedEmbedder() == nil)
        await store.close()
    }

    @Test func garbageFileIsReportedAsCorrupted() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(repeating: 0x41, count: 4096).write(to: directory.appendingPathComponent(Store.databaseName))
        #expect(throws: StoreError.self) {
            _ = try Store(directory: directory, fileProtection: .completeUntilFirstUserAuthentication)
        }
    }
}

@Suite("FTS query builder")
struct FTSQueryTests {
    @Test func quotesDedupesAndOrsTerms() {
        #expect(FTSQuery.build("What is the Sample-size? sample") == "\"what\" OR \"is\" OR \"the\" OR \"sample\" OR \"size\"")
    }

    @Test func returnsNilWithoutTerms() {
        #expect(FTSQuery.build("?!  ") == nil)
    }

    @Test func capsTermCount() {
        let query = (1...200).map { "w\($0)" }.joined(separator: " ")
        #expect(FTSQuery.build(query)?.components(separatedBy: " OR ").count == FTSQuery.maxTerms)
    }
}
