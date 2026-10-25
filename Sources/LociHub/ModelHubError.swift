import Foundation

/// Why a model download failed.
public enum ModelHubError: Error, Sendable, Equatable, LocalizedError {
    /// The source is not a Hugging Face file (`.hub`), or its repo or file name is malformed.
    case invalidSource(String)
    /// The server answered with an error status (404 for a missing file, 401/403 for a gated repo).
    case httpStatus(Int)
    /// The downloaded file does not match the expected SHA-256; it has been deleted.
    case checksumMismatch(expected: String, actual: String)
    case insufficientDiskSpace(required: Int64, available: Int64)
    case network(String)
    case io(String)

    public var errorDescription: String? {
        switch self {
        case .invalidSource(let reason): "This model cannot be downloaded: \(reason)."
        case .httpStatus(404): "The model file was not found on the server."
        case .httpStatus(let code) where code == 401 || code == 403: "The server refused the download (HTTP \(code)); the repository may require a licence acceptance or login."
        case .httpStatus(let code): "The server answered with HTTP \(code)."
        case .checksumMismatch: "The downloaded file is corrupt (its checksum does not match). It was deleted; try again."
        case .insufficientDiskSpace(let required, let available):
            "Not enough free space: \(ByteCountFormatter.string(fromByteCount: required, countStyle: .file)) needed, "
                + "\(ByteCountFormatter.string(fromByteCount: available, countStyle: .file)) available."
        case .network(let reason): "The download failed: \(reason)."
        case .io(let reason): "The model file could not be written: \(reason)."
        }
    }
}

/// How far a download has got.
public struct DownloadProgress: Sendable, Equatable {
    /// Bytes on disk so far, including any part kept from an earlier attempt.
    public let receivedBytes: Int64
    /// The full size, when the server said.
    public let totalBytes: Int64?

    public var fraction: Double? {
        guard let totalBytes, totalBytes > 0 else { return nil }
        return min(Double(receivedBytes) / Double(totalBytes), 1)
    }
}

public enum DownloadEvent: Sendable, Equatable {
    case progress(DownloadProgress)
    /// The verified file is in place at this location.
    case finished(URL)
}
