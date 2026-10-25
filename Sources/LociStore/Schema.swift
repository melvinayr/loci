/// Schema creation and versioned migrations, tracked in `meta.schema_version`.
enum Schema {
    static let currentVersion = 2

    /// `migrations[n]` upgrades version `n` to `n + 1`. A fresh database starts at version 0.
    private static let migrations: [@Sendable (SQLiteConnection) throws -> Void] = [
        { db in
            for statement in [
                """
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY NOT NULL,
                    value TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE documents (
                    id TEXT PRIMARY KEY NOT NULL,
                    title TEXT NOT NULL,
                    uri TEXT,
                    kind TEXT NOT NULL,
                    page_count INTEGER,
                    content_hash TEXT NOT NULL UNIQUE,
                    metadata_json TEXT NOT NULL,
                    added_at REAL NOT NULL,
                    text TEXT NOT NULL,
                    pages_json TEXT
                )
                """,
                "CREATE INDEX documents_uri ON documents(uri)",
                """
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    page INTEGER,
                    start INTEGER NOT NULL,
                    "end" INTEGER NOT NULL,
                    heading_path TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    text TEXT NOT NULL
                )
                """,
                "CREATE INDEX chunks_document ON chunks(document_id, ordinal)",
                """
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    text, content='chunks', content_rowid='id',
                    tokenize='porter unicode61 remove_diacritics 2'
                )
                """,
                """
                CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
                END
                """,
                """
                CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
                END
                """,
            ] {
                try db.execute(statement, cached: false)
            }
        },
        // v2: half-precision embeddings, one row per chunk, removed with their chunk.
        { db in
            try db.execute(
                """
                CREATE TABLE vectors (
                    chunk_id INTEGER PRIMARY KEY NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    embedding BLOB NOT NULL
                )
                """,
                cached: false)
        },
    ]

    /// Brings the database to `currentVersion`, refusing databases from a newer release.
    /// `target` lets tests build a database at an older version.
    static func migrate(_ db: SQLiteConnection, to target: Int = currentVersion) throws {
        guard SQLiteCapabilities.hasFTS5() else {
            throw StoreError.corrupted("this system's SQLite lacks FTS5")
        }
        var version = try storedVersion(db)
        guard version <= currentVersion else {
            throw StoreError.corrupted("the store was written by a newer version of Loci")
        }
        while version < min(target, currentVersion) {
            try db.transaction {
                try migrations[version](db)
                try db.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?1) "
                        + "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    [.text(String(version + 1))], cached: false)
            }
            version += 1
        }
    }

    static func storedVersion(_ db: SQLiteConnection) throws -> Int {
        var hasMeta = false
        try db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'", cached: false) { _ in
            hasMeta = true
        }
        guard hasMeta else { return 0 }
        var version = 0
        try db.execute("SELECT value FROM meta WHERE key = 'schema_version'", cached: false) {
            version = Int($0.text(0)) ?? 0
        }
        return version
    }
}
