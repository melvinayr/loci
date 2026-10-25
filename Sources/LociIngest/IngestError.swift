/// Failures raised while parsing or chunking. `Loci` converts these to `LociError`.
public enum IngestError: Error, Sendable, Equatable {
    case unsupported(reason: String)
    case unreadable(reason: String)
}
