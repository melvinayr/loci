import SQLite3

/// A single SQLite connection with a prepared-statement cache.
///
/// Not thread-safe: each instance is owned by exactly one actor (`StoreWriter` or
/// `StoreReader`), which serialises all access. Statements are finalised and the
/// handle closed in `deinit`.
final class SQLiteConnection {
    enum Bind {
        case int(Int64)
        case double(Double)
        case text(String)
        case blob([UInt16])
        case null
    }

    struct Row {
        let statement: OpaquePointer

        func int(_ column: Int32) -> Int64 { sqlite3_column_int64(statement, column) }
        func double(_ column: Int32) -> Double { sqlite3_column_double(statement, column) }
        func isNull(_ column: Int32) -> Bool { sqlite3_column_type(statement, column) == SQLITE_NULL }
        func text(_ column: Int32) -> String {
            sqlite3_column_text(statement, column).map { String(cString: $0) } ?? ""
        }
        /// The raw bytes of a BLOB column; valid until the row is advanced.
        func blob(_ column: Int32) -> UnsafeRawBufferPointer {
            guard let bytes = sqlite3_column_blob(statement, column) else { return UnsafeRawBufferPointer(start: nil, count: 0) }
            return UnsafeRawBufferPointer(start: bytes, count: Int(sqlite3_column_bytes(statement, column)))
        }
        func optionalInt(_ column: Int32) -> Int? { isNull(column) ? nil : Int(int(column)) }
        func optionalText(_ column: Int32) -> String? { isNull(column) ? nil : text(column) }
    }

    private var handle: OpaquePointer?
    private var cache: [String: OpaquePointer] = [:]

    init(path: String, readOnly: Bool) throws {
        let flags = (readOnly ? SQLITE_OPEN_READONLY : SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE)
            | SQLITE_OPEN_NOMUTEX
        var opened: OpaquePointer?
        let code = sqlite3_open_v2(path, &opened, flags, nil)
        guard code == SQLITE_OK, let opened else {
            sqlite3_close(opened)
            throw Self.error("open", code)
        }
        handle = opened
        sqlite3_busy_timeout(opened, 5000)
        try execute("PRAGMA foreign_keys = ON", cached: false)
        if !readOnly {
            try execute("PRAGMA journal_mode = WAL", cached: false)
            try execute("PRAGMA synchronous = NORMAL", cached: false)
        }
    }

    deinit {
        close()
    }

    /// Finalises statements and closes the database. Idempotent.
    func close() {
        for statement in cache.values { sqlite3_finalize(statement) }
        cache.removeAll()
        if let handle { sqlite3_close_v2(handle) }
        handle = nil
    }

    var lastInsertRowID: Int64 { handle.map(sqlite3_last_insert_rowid) ?? 0 }
    var changes: Int { handle.map { Int(sqlite3_changes($0)) } ?? 0 }

    /// Runs one statement, calling `row` for each result row.
    func execute(
        _ sql: String,
        _ binds: [Bind] = [],
        cached: Bool = true,
        row: ((Row) throws -> Void)? = nil
    ) throws {
        let statement = try prepare(sql, cached: cached)
        defer {
            if cached {
                sqlite3_reset(statement)
                sqlite3_clear_bindings(statement)
            } else {
                sqlite3_finalize(statement)
            }
        }
        for (offset, bind) in binds.enumerated() {
            let index = Int32(offset + 1)
            let code: Int32
            switch bind {
            case .int(let value): code = sqlite3_bind_int64(statement, index, value)
            case .double(let value): code = sqlite3_bind_double(statement, index, value)
            case .text(let value): code = sqlite3_bind_text(statement, index, value, -1, Self.transient)
            case .blob(let value):
                code = value.withUnsafeBytes { sqlite3_bind_blob(statement, index, $0.baseAddress, Int32($0.count), Self.transient) }
            case .null: code = sqlite3_bind_null(statement, index)
            }
            guard code == SQLITE_OK else { throw Self.error("bind", code) }
        }
        while true {
            let code = sqlite3_step(statement)
            if code == SQLITE_ROW {
                try row?(Row(statement: statement))
            } else if code == SQLITE_DONE {
                return
            } else {
                throw Self.error("step", code)
            }
        }
    }

    /// Runs `body` in an immediate transaction, rolling back if it throws.
    func transaction<T>(_ body: () throws -> T) throws -> T {
        try execute("BEGIN IMMEDIATE", cached: false)
        do {
            let result = try body()
            try execute("COMMIT", cached: false)
            return result
        } catch {
            try? execute("ROLLBACK", cached: false)
            throw error
        }
    }

    // MARK: Private

    private func prepare(_ sql: String, cached: Bool) throws -> OpaquePointer {
        if cached, let statement = cache[sql] { return statement }
        guard let handle else { throw StoreError.corrupted("the store is closed") }
        var statement: OpaquePointer?
        let code = sqlite3_prepare_v2(handle, sql, -1, &statement, nil)
        guard code == SQLITE_OK, let statement else {
            sqlite3_finalize(statement)
            throw Self.error("prepare", code)
        }
        if cached { cache[sql] = statement }
        return statement
    }

    /// `SQLITE_TRANSIENT`: SQLite copies the bound text before `bind` returns.
    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    private static func error(_ operation: String, _ code: Int32) -> StoreError {
        switch code & 0xFF {
        case SQLITE_NOTADB, SQLITE_CORRUPT:
            return .corrupted("the database file is not a valid Loci store")
        default:
            return .sqlite(operation: operation, code: code)
        }
    }
}
