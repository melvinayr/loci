import Foundation
internal import LociIngest
internal import LociLlama
internal import LociStore

/// Errors thrown by Loci. Messages never contain document text, queries or answers.
public enum LociError: Error, Sendable, LocalizedError {
    case unsupportedDocument(URL, reason: String)
    /// Encrypted PDF, scanned PDF with no text layer, or a file that cannot be read.
    case documentUnreadable(URL, reason: String)
    case modelNotFound(ModelSource)
    case invalidModel(ModelSource, reason: String)
    /// For example a chat model passed as the embedder.
    case modelRoleMismatch(ModelSource, expected: String)
    case embedderMismatch(stored: String, configured: String)
    case insufficientMemory(required: UInt64, available: UInt64)
    /// `ask()` on a retrieval-only knowledge base.
    case noGenerator
    case storeCorrupted(reason: String)
    case backendFailure(operation: String, code: Int32)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .unsupportedDocument(let url, let reason):
            "“\(url.lastPathComponent)” is not supported: \(reason)."
        case .documentUnreadable(let url, let reason):
            "“\(url.lastPathComponent)” could not be read: \(reason)."
        case .modelNotFound(let source):
            "The model file \(source.displayName) could not be found."
        case .invalidModel(_, let reason):
            "The model is not valid: \(reason)."
        case .modelRoleMismatch(_, let expected):
            "The model cannot be used here; a \(expected) model is required."
        case .embedderMismatch:
            "This knowledge base was built with a different embedding model. Call reindex() to rebuild it."
        case .insufficientMemory(let required, let available):
            "Not enough memory: \(required) bytes needed, \(available) available."
        case .noGenerator:
            "This knowledge base has no generator model, so it can only search."
        case .storeCorrupted(let reason):
            "The knowledge base store is unusable: \(reason)."
        case .backendFailure(let operation, let code):
            "\(operation) failed (code \(code))."
        case .cancelled:
            "The operation was cancelled."
        }
    }

    /// Converts any lower-layer error to a `LociError`.
    static func from(_ error: Error, url: URL? = nil) -> LociError {
        switch error {
        case let error as LociError:
            return error
        case is CancellationError:
            return .cancelled
        case let error as StoreError:
            switch error {
            case .embedderMismatch(let stored, let configured): return .embedderMismatch(stored: stored, configured: configured)
            case .corrupted(let reason): return .storeCorrupted(reason: reason)
            case .sqlite(let operation, let code): return .backendFailure(operation: "Store \(operation)", code: code)
            }
        case let error as IngestError:
            let location = url ?? URL(fileURLWithPath: "/")
            switch error {
            case .unsupported(let reason): return .unsupportedDocument(location, reason: reason)
            case .unreadable(let reason): return .documentUnreadable(location, reason: reason)
            }
        case let error as LlamaError:
            switch error {
            case .backend(let operation, let code): return .backendFailure(operation: "Model \(operation)", code: code)
            case .insufficientMemory(let required, let available): return .insufficientMemory(required: required, available: available)
            case .fileNotFound, .invalidModel, .roleMismatch: return .backendFailure(operation: "Model loading", code: -1)
            }
        default:
            return .backendFailure(operation: "Unexpected error", code: -1)
        }
    }

    /// Like `from(_:url:)`, attributing model-loading failures to `model`.
    static func from(_ error: Error, model: ModelSource) -> LociError {
        guard let error = error as? LlamaError else { return from(error) }
        switch error {
        case .fileNotFound: return .modelNotFound(model)
        case .invalidModel(let reason): return .invalidModel(model, reason: reason)
        case .roleMismatch(let expected): return .modelRoleMismatch(model, expected: expected)
        case .backend, .insufficientMemory: return from(error)
        }
    }
}
