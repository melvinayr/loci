/// Failures raised by the store. `Loci` converts these to `LociError`.
public enum StoreError: Error, Sendable, Equatable {
    /// A SQLite call failed; `code` is the SQLite result code.
    case sqlite(operation: String, code: Int32)
    /// The database is damaged, from a newer version, or otherwise unusable.
    case corrupted(String)
    /// The store's vectors were built with a different embedding model than the one supplied.
    case embedderMismatch(stored: String, configured: String)
}
