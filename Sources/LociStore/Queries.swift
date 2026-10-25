import Foundation

/// Read queries shared by the writer and reader connections.
extension SQLiteConnection {
    private static let documentColumns = """
        d.id, d.title, d.uri, d.kind, d.page_count, d.content_hash, d.metadata_json, d.added_at,
        (SELECT COUNT(*) FROM chunks WHERE document_id = d.id)
        """
    static let chunkColumns = "c.id, c.document_id, c.ordinal, c.page, c.start, c.\"end\", c.heading_path, c.token_count, c.text"

    private static func document(_ row: Row) -> StoredDocument {
        StoredDocument(
            id: row.text(0), title: row.text(1), uri: row.optionalText(2), kind: row.text(3),
            pageCount: row.optionalInt(4), chunkCount: Int(row.int(8)), contentHash: row.text(5),
            metadata: decode(row.text(6)) ?? [:],
            addedAt: Date(timeIntervalSince1970: row.double(7)))
    }

    private static func chunk(_ row: Row, offset: Int32 = 0) -> StoredChunk {
        StoredChunk(
            id: row.int(offset), documentID: row.text(offset + 1), ordinal: Int(row.int(offset + 2)),
            page: row.optionalInt(offset + 3), start: Int(row.int(offset + 4)), end: Int(row.int(offset + 5)),
            headingPath: decode(row.text(offset + 6)) ?? [], tokenCount: Int(row.int(offset + 7)),
            text: row.text(offset + 8))
    }

    private static func decode<T: Decodable>(_ json: String) -> T? {
        try? JSONDecoder().decode(T.self, from: Data(json.utf8))
    }

    func allDocuments() throws -> [StoredDocument] {
        var result: [StoredDocument] = []
        try execute("SELECT \(Self.documentColumns) FROM documents d ORDER BY d.added_at, d.id") {
            result.append(Self.document($0))
        }
        return result
    }

    func document(id: String) throws -> StoredDocument? {
        var result: StoredDocument?
        try execute("SELECT \(Self.documentColumns) FROM documents d WHERE d.id = ?1", [.text(id)]) {
            result = Self.document($0)
        }
        return result
    }

    func document(hash: String) throws -> StoredDocument? {
        var result: StoredDocument?
        try execute("SELECT \(Self.documentColumns) FROM documents d WHERE d.content_hash = ?1", [.text(hash)]) {
            result = Self.document($0)
        }
        return result
    }

    func documentText(id: String) throws -> String? {
        var result: String?
        try execute("SELECT text FROM documents WHERE id = ?1", [.text(id)]) { result = $0.text(0) }
        return result
    }

    func pageRanges(id: String) throws -> [Range<Int>?]? {
        var json: String?
        try execute("SELECT pages_json FROM documents WHERE id = ?1", [.text(id)]) { json = $0.optionalText(0) }
        guard let json, let pairs: [[Int]?] = Self.decode(json) else { return nil }
        return pairs.map { pair in
            guard let pair, pair.count == 2, pair[0] <= pair[1] else { return nil }
            return pair[0]..<pair[1]
        }
    }

    func chunks(documentID: String) throws -> [StoredChunk] {
        var result: [StoredChunk] = []
        try execute(
            "SELECT \(Self.chunkColumns) FROM chunks c WHERE c.document_id = ?1 ORDER BY c.ordinal",
            [.text(documentID)]
        ) { result.append(Self.chunk($0)) }
        return result
    }

    func chunk(id: Int64) throws -> StoredChunk? {
        var result: StoredChunk?
        try execute("SELECT \(Self.chunkColumns) FROM chunks c WHERE c.id = ?1", [.int(id)]) {
            result = Self.chunk($0)
        }
        return result
    }

    func keywordSearch(match: String, limit: Int, documentIDs: [String]?) throws -> [KeywordHit] {
        var sql = """
            SELECT \(Self.chunkColumns), bm25(chunks_fts)
            FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?1
            """
        var binds: [Bind] = [.text(match)]
        if let documentIDs {
            sql += " AND c.document_id IN (" + documentIDs.indices.map { "?\($0 + 2)" }.joined(separator: ",") + ")"
            binds += documentIDs.map { .text($0) }
        }
        sql += " ORDER BY bm25(chunks_fts) LIMIT ?\(binds.count + 1)"
        binds.append(.int(Int64(limit)))

        var hits: [KeywordHit] = []
        try execute(sql, binds, cached: documentIDs == nil) { row in
            hits.append(KeywordHit(chunk: Self.chunk(row), score: -row.double(9)))
        }
        return hits
    }

    func counts() throws -> (documents: Int, chunks: Int, vectors: Int) {
        var result = (documents: 0, chunks: 0, vectors: 0)
        try execute(
            "SELECT (SELECT COUNT(*) FROM documents), (SELECT COUNT(*) FROM chunks), (SELECT COUNT(*) FROM vectors)"
        ) {
            result = (Int($0.int(0)), Int($0.int(1)), Int($0.int(2)))
        }
        return result
    }

    /// Chunks by ID, in the order requested; unknown IDs are skipped.
    func chunks(ids: [Int64]) throws -> [StoredChunk] {
        var found: [Int64: StoredChunk] = [:]
        // Stay well under SQLite's bound-variable limit on older systems.
        for slice in stride(from: 0, to: ids.count, by: 500).map({ Array(ids[$0..<min($0 + 500, ids.count)]) }) {
            let placeholders = slice.indices.map { "?\($0 + 1)" }.joined(separator: ",")
            try execute(
                "SELECT \(Self.chunkColumns) FROM chunks c WHERE c.id IN (\(placeholders))",
                slice.map { .int($0) }, cached: false
            ) { row in
                let chunk = Self.chunk(row)
                found[chunk.id] = chunk
            }
        }
        return ids.compactMap { found[$0] }
    }

    /// Every stored vector, skipping rows whose size does not match `dimension`
    /// (left over from an interrupted reindex).
    func vectorSnapshot(dimension: Int) throws -> VectorSnapshot {
        var snapshot = VectorSnapshot()
        snapshot.dimension = dimension
        var ordinals: [String: Int32] = [:]
        try execute(
            "SELECT v.chunk_id, c.document_id, v.embedding FROM vectors v JOIN chunks c ON c.id = v.chunk_id ORDER BY v.chunk_id"
        ) { row in
            let bytes = row.blob(2)
            guard bytes.count == dimension * MemoryLayout<UInt16>.stride else { return }
            let documentID = row.text(1)
            let ordinal: Int32
            if let known = ordinals[documentID] {
                ordinal = known
            } else {
                ordinal = Int32(snapshot.documentIDs.count)
                ordinals[documentID] = ordinal
                snapshot.documentIDs.append(documentID)
            }
            snapshot.chunkIDs.append(row.int(0))
            snapshot.documentOrdinals.append(ordinal)
            snapshot.components.append(contentsOf: bytes.bindMemory(to: UInt16.self))
        }
        return snapshot
    }

    func meta(_ key: String) throws -> String? {
        var result: String?
        try execute("SELECT value FROM meta WHERE key = ?1", [.text(key)]) { result = $0.text(0) }
        return result
    }
}
