import SQLite3

/// Checks of the system SQLite that the store depends on.
enum SQLiteCapabilities {
    /// True when the linked SQLite was built with FTS5 and BM25 ranking works.
    static func hasFTS5() -> Bool {
        var db: OpaquePointer?
        guard sqlite3_open(":memory:", &db) == SQLITE_OK, let db else {
            sqlite3_close(db)
            return false
        }
        defer { sqlite3_close(db) }
        let sql = """
        CREATE VIRTUAL TABLE probe USING fts5(text);
        INSERT INTO probe(text) VALUES ('probe');
        SELECT bm25(probe) FROM probe WHERE probe MATCH 'probe';
        """
        return sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK
    }

    /// Output of `PRAGMA compile_options`, for diagnostics.
    static func compileOptions() -> [String] {
        var db: OpaquePointer?
        guard sqlite3_open(":memory:", &db) == SQLITE_OK, let db else {
            sqlite3_close(db)
            return []
        }
        defer { sqlite3_close(db) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, "PRAGMA compile_options", -1, &statement, nil) == SQLITE_OK else {
            return []
        }
        defer { sqlite3_finalize(statement) }
        var options: [String] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            if let text = sqlite3_column_text(statement, 0) {
                options.append(String(cString: text))
            }
        }
        return options
    }
}
