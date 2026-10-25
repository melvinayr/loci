/// Failures raised by llama.cpp wrappers. `Loci` converts these to `LociError`.
public enum LlamaError: Error, Sendable, Equatable {
    case fileNotFound
    /// The file could not be loaded as a model.
    case invalidModel(reason: String)
    /// The model loaded but cannot play the requested role (for example a chat model as embedder).
    case roleMismatch(expected: String)
    /// Loading would exceed what the process may use; nothing was loaded.
    case insufficientMemory(required: UInt64, available: UInt64)
    case backend(operation: String, code: Int32)
}
