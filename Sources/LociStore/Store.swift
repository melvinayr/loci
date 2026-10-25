import Foundation
import os

/// The SQLite store for one knowledge base: documents, chunks and full-text index.
///
/// One writer connection and one WAL reader connection, each confined to its own actor,
/// so searches run concurrently with ingest. Every document is written in a single
/// transaction.
public final class Store: Sendable {
    public static let databaseName = "loci.sqlite"
    static let embedderIDKey = "embedder_id"
    static let embedderDimensionKey = "embedder_dimension"

    /// Documents above this many IDs are filtered after the query instead of in SQL, to
    /// stay under SQLite's bound-variable limit on older systems.
    static let inlineFilterLimit = 900
    static let postFilterFetchLimit = 5_000

    public let directory: URL
    private let writer: StoreWriter
    private let reader: StoreReader
    private let vectors = VectorCache()

    /// Opens or creates the store in `directory`, migrating the schema if needed.
    public init(directory: URL, fileProtection: FileProtectionType) throws {
        self.directory = directory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileProtection.apply(fileProtection, to: directory)

        let database = directory.appendingPathComponent(Self.databaseName)
        writer = try StoreWriter(databaseURL: database, fileProtection: fileProtection)
        reader = try StoreReader(databaseURL: database)
    }

    /// Files SQLite keeps for the database, all of which receive file protection.
    public static func managedFileNames() -> [String] {
        [databaseName, databaseName + "-wal", databaseName + "-shm"]
    }

    // MARK: Write

    /// Stores a document, its chunks and (when `embedder` is given) one vector per chunk, in one
    /// transaction. Throws `StoreError.embedderMismatch` if the store holds vectors from another model.
    public func insert(
        _ document: NewDocument, chunks: [NewChunk], vectors newVectors: [[Float]] = [], embedder: StoreEmbedder? = nil
    ) async throws -> InsertOutcome {
        let outcome = try await writer.insert(document, chunks: chunks, vectors: newVectors, embedder: embedder)
        await self.vectors.invalidate()
        return outcome
    }

    /// Replaces the vectors of one document (`vectors[i]` belongs to the chunk with ordinal `i`).
    public func replaceVectors(documentID: String, vectors newVectors: [[Float]], embedder: StoreEmbedder) async throws {
        try await writer.replaceVectors(documentID: documentID, vectors: newVectors, embedder: embedder)
        await self.vectors.invalidate()
    }

    /// Records which embedding model the vectors were built with, once a reindex has finished.
    public func setEmbedder(_ embedder: StoreEmbedder) async throws {
        try await writer.setEmbedder(embedder)
        await self.vectors.invalidate()
    }

    /// Removes a document and its chunks. Returns false if it did not exist.
    @discardableResult
    public func remove(documentID: String) async throws -> Bool {
        let removed = try await writer.remove(documentID: documentID)
        await self.vectors.invalidate()
        return removed
    }

    public func setMeta(_ key: String, _ value: String) async throws {
        try await writer.setMeta(key, value)
    }

    // MARK: Read

    public func documents() async throws -> [StoredDocument] { try await reader.run { try $0.allDocuments() } }
    public func document(id: String) async throws -> StoredDocument? { try await reader.run { try $0.document(id: id) } }
    public func document(hash: String) async throws -> StoredDocument? { try await reader.run { try $0.document(hash: hash) } }
    public func documentText(id: String) async throws -> String? { try await reader.run { try $0.documentText(id: id) } }
    public func pageRanges(documentID: String) async throws -> [Range<Int>?]? {
        try await reader.run { try $0.pageRanges(id: documentID) }
    }
    public func chunks(documentID: String) async throws -> [StoredChunk] {
        try await reader.run { try $0.chunks(documentID: documentID) }
    }
    public func chunk(id: Int64) async throws -> StoredChunk? { try await reader.run { try $0.chunk(id: id) } }
    public func meta(_ key: String) async throws -> String? { try await reader.run { try $0.meta(key) } }
    public func chunks(ids: [Int64]) async throws -> [StoredChunk] { try await reader.run { try $0.chunks(ids: ids) } }

    /// The embedding model the stored vectors belong to, or nil before the first vector is stored.
    public func storedEmbedder() async throws -> StoreEmbedder? {
        try await reader.run { db in
            guard let id = try db.meta(Self.embedderIDKey), let raw = try db.meta(Self.embedderDimensionKey),
                  let dimension = Int(raw) else { return nil }
            return StoreEmbedder(id: id, dimension: dimension)
        }
    }

    /// Nearest chunks to `query` (a unit vector from the stored embedder) by cosine similarity.
    public func vectorSearch(_ query: [Float], limit: Int, documentIDs: Set<String>? = nil) async throws -> [VectorHit] {
        guard limit > 0, !query.isEmpty else { return [] }
        if let documentIDs, documentIDs.isEmpty { return [] }
        let dimension = query.count
        let reader = self.reader
        return try await vectors.search(query, limit: limit, allowedDocuments: documentIDs) {
            try await reader.run { try $0.vectorSnapshot(dimension: dimension) }
        }
    }

    /// BM25 search over chunk text. `documentIDs` restricts results to those documents.
    public func keywordSearch(_ query: String, limit: Int, documentIDs: Set<String>? = nil) async throws -> [KeywordHit] {
        try await keywordSearch(query, limit: limit, documentIDs: documentIDs, inlineFilterLimit: Self.inlineFilterLimit)
    }

    func keywordSearch(
        _ query: String, limit: Int, documentIDs: Set<String>?, inlineFilterLimit: Int
    ) async throws -> [KeywordHit] {
        guard limit > 0, let match = FTSQuery.build(query) else { return [] }
        if let documentIDs {
            if documentIDs.isEmpty { return [] }
            if documentIDs.count > inlineFilterLimit {
                let hits = try await reader.run {
                    try $0.keywordSearch(match: match, limit: Self.postFilterFetchLimit, documentIDs: nil)
                }
                return Array(hits.filter { documentIDs.contains($0.chunk.documentID) }.prefix(limit))
            }
            return try await reader.run {
                try $0.keywordSearch(match: match, limit: limit, documentIDs: documentIDs.sorted())
            }
        }
        return try await reader.run { try $0.keywordSearch(match: match, limit: limit, documentIDs: nil) }
    }

    public func stats() async throws -> StoreStats {
        let counts = try await reader.run { try $0.counts() }
        let manager = FileManager.default
        let bytes = Self.managedFileNames().reduce(UInt64(0)) { total, name in
            let size = (try? manager.attributesOfItem(atPath: directory.appendingPathComponent(name).path)[.size]) as? NSNumber
            return total + (size?.uint64Value ?? 0)
        }
        return StoreStats(
            documentCount: counts.documents, chunkCount: counts.chunks, vectorCount: counts.vectors, databaseBytes: bytes)
    }

    /// Closes both connections. Further calls throw.
    public func close() async {
        await reader.close()
        await writer.close()
    }
}

// MARK: - Actors

private let log = Logger(subsystem: "dev.loci", category: "store")

actor StoreWriter {
    private let connection: SQLiteConnection
    private let databaseURL: URL
    private let fileProtection: FileProtectionType

    init(databaseURL: URL, fileProtection: FileProtectionType) throws {
        let connection = try SQLiteConnection(path: databaseURL.path, readOnly: false)
        try Schema.migrate(connection)
        self.connection = connection
        self.databaseURL = databaseURL
        self.fileProtection = fileProtection
        try Self.protectFiles(at: databaseURL, with: fileProtection)
    }

    func insert(
        _ document: NewDocument, chunks: [NewChunk], vectors: [[Float]], embedder: StoreEmbedder?
    ) throws -> InsertOutcome {
        if let embedder {
            guard vectors.count == chunks.count, vectors.allSatisfy({ $0.count == embedder.dimension }) else {
                throw StoreError.corrupted("vectors do not match the chunks or the embedder dimension")
            }
        }
        let outcome = try connection.transaction { () throws -> InsertOutcome in
            if let existing = try connection.document(hash: document.contentHash) {
                return .existing(existing)
            }
            if let embedder { try claimEmbedder(embedder) }
            if let uri = document.uri {
                // The same source with changed content replaces its earlier version.
                try connection.execute("DELETE FROM documents WHERE uri = ?1", [.text(uri)])
            }

            let id = UUID().uuidString
            // Round-trip through the stored double so the returned value equals what is read back.
            let addedAt = Date(timeIntervalSince1970: Date().timeIntervalSince1970)
            try connection.execute(
                """
                INSERT INTO documents(id, title, uri, kind, page_count, content_hash, metadata_json, added_at, text, pages_json)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                """,
                [
                    .text(id), .text(document.title), document.uri.map { .text($0) } ?? .null, .text(document.kind),
                    document.pageCount.map { .int(Int64($0)) } ?? .null, .text(document.contentHash),
                    .text(Self.json(document.metadata)), .double(addedAt.timeIntervalSince1970), .text(document.text),
                    document.pageRanges.map { .text(Self.json($0.map { $0.map { [$0.lowerBound, $0.upperBound] } })) } ?? .null,
                ])
            for (index, chunk) in chunks.enumerated() {
                try connection.execute(
                    """
                    INSERT INTO chunks(document_id, ordinal, page, start, "end", heading_path, token_count, text)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                    """,
                    [
                        .text(id), .int(Int64(chunk.ordinal)), chunk.page.map { .int(Int64($0)) } ?? .null,
                        .int(Int64(chunk.start)), .int(Int64(chunk.end)), .text(Self.json(chunk.headingPath)),
                        .int(Int64(chunk.tokenCount)), .text(chunk.text),
                    ])
                if embedder != nil {
                    try connection.execute(
                        "INSERT INTO vectors(chunk_id, embedding) VALUES (?1, ?2)",
                        [.int(connection.lastInsertRowID), .blob(Float16Codec.encode(vectors[index]))])
                }
            }
            let stored = StoredDocument(
                id: id, title: document.title, uri: document.uri, kind: document.kind,
                pageCount: document.pageCount, chunkCount: chunks.count, contentHash: document.contentHash,
                metadata: document.metadata, addedAt: addedAt)
            return .inserted(stored)
        }
        try protectFiles()
        if case .inserted(let stored) = outcome {
            log.info("stored document \(stored.id, privacy: .private) with \(stored.chunkCount) chunks")
        }
        return outcome
    }

    func remove(documentID: String) throws -> Bool {
        try connection.execute("DELETE FROM documents WHERE id = ?1", [.text(documentID)])
        let removed = connection.changes > 0
        try protectFiles()
        return removed
    }

    func replaceVectors(documentID: String, vectors: [[Float]], embedder: StoreEmbedder) throws {
        guard vectors.allSatisfy({ $0.count == embedder.dimension }) else {
            throw StoreError.corrupted("vectors do not match the embedder dimension")
        }
        try connection.transaction {
            var chunkIDs: [Int64] = []
            try connection.execute("SELECT id FROM chunks WHERE document_id = ?1 ORDER BY ordinal", [.text(documentID)]) {
                chunkIDs.append($0.int(0))
            }
            guard chunkIDs.count == vectors.count else {
                throw StoreError.corrupted("vector count does not match the document's chunks")
            }
            for (chunkID, vector) in zip(chunkIDs, vectors) {
                try connection.execute(
                    "INSERT INTO vectors(chunk_id, embedding) VALUES (?1, ?2) "
                        + "ON CONFLICT(chunk_id) DO UPDATE SET embedding = excluded.embedding",
                    [.int(chunkID), .blob(Float16Codec.encode(vector))])
            }
        }
        try protectFiles()
    }

    func setEmbedder(_ embedder: StoreEmbedder) throws {
        try setMeta(Store.embedderIDKey, embedder.id)
        try setMeta(Store.embedderDimensionKey, String(embedder.dimension))
    }

    /// Records `embedder` as the store's model on first use; later calls must match it.
    /// Runs inside the caller's transaction.
    private func claimEmbedder(_ embedder: StoreEmbedder) throws {
        let stored = try connection.meta(Store.embedderIDKey)
        guard let stored else {
            try setEmbedder(embedder)
            return
        }
        guard stored == embedder.id else {
            throw StoreError.embedderMismatch(stored: stored, configured: embedder.id)
        }
        // The ID can be adopted from a file header before the model's true output size is known.
        if try connection.meta(Store.embedderDimensionKey) != String(embedder.dimension) {
            try setEmbedder(embedder)
        }
    }

    func setMeta(_ key: String, _ value: String) throws {
        try connection.execute(
            "INSERT INTO meta(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [.text(key), .text(value)])
    }

    func close() {
        connection.close()
    }

    /// SQLite creates the WAL and SHM files lazily, so protection is re-applied after writes.
    private func protectFiles() throws {
        try Self.protectFiles(at: databaseURL, with: fileProtection)
    }

    private static func protectFiles(at databaseURL: URL, with type: FileProtectionType) throws {
        let directory = databaseURL.deletingLastPathComponent()
        for name in Store.managedFileNames() {
            try FileProtection.apply(type, to: directory.appendingPathComponent(name))
        }
    }

    private static func json<T: Encodable>(_ value: T) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return (try? encoder.encode(value)).flatMap { String(data: $0, encoding: .utf8) } ?? "null"
    }
}

actor StoreReader {
    private let connection: SQLiteConnection

    init(databaseURL: URL) throws {
        connection = try SQLiteConnection(path: databaseURL.path, readOnly: true)
    }

    func run<T: Sendable>(_ body: @Sendable (SQLiteConnection) throws -> T) throws -> T {
        try body(connection)
    }

    func close() {
        connection.close()
    }
}
