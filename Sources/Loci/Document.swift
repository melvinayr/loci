import Foundation

/// Identifies one document within a knowledge base.
public struct DocumentID: Hashable, Sendable, Codable, RawRepresentable, CustomStringConvertible {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public init(from decoder: Decoder) throws {
        rawValue = try decoder.singleValueContainer().decode(String.self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    public var description: String { rawValue }
}

/// Identifies one chunk (passage) within a knowledge base.
public struct ChunkID: Hashable, Sendable, Codable, RawRepresentable, CustomStringConvertible {
    public let rawValue: Int64

    public init(rawValue: Int64) {
        self.rawValue = rawValue
    }

    public init(from decoder: Decoder) throws {
        rawValue = try decoder.singleValueContainer().decode(Int64.self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    public var description: String { String(rawValue) }
}

/// The format a document was ingested from.
public enum DocumentKind: String, Sendable, Codable, Hashable {
    case pdf, markdown, html, text
}

/// Where a document comes from.
public enum DocumentSource: Sendable {
    /// pdf, md, markdown, html, htm or txt.
    case file(URL)
    case text(String, title: String, uri: String? = nil)
}

/// A document stored in a knowledge base.
public struct Document: Sendable, Identifiable, Hashable, Codable {
    /// Stable identifier, generated when the document is first stored.
    public let id: DocumentID
    /// The document title, or the file name when the format has none.
    public let title: String
    /// The source location (file URL string) or the caller-supplied URI.
    public let uri: String?
    /// The format the document was ingested from.
    public let kind: DocumentKind
    /// Number of pages, PDFs only.
    public let pageCount: Int?
    /// Number of passages the document was split into.
    public let chunkCount: Int
    /// SHA-256 of the normalised text; re-adding an unchanged document is a no-op.
    public let contentHash: String
    /// When the document was stored.
    public let addedAt: Date
}

/// Progress of an ingest operation.
public struct IngestProgress: Sendable, Equatable {
    /// A step in ingesting one document. `.skipped` carries a human-readable reason.
    public enum Phase: Sendable, Equatable {
        case parsing, chunking, embedding, storing, done
        case skipped(String)
    }

    /// File name or title of the document being processed.
    public let document: String
    /// The step just started or finished for `document`.
    public let phase: Phase
    /// Documents finished so far in this operation.
    public let completed: Int
    /// Documents in this operation.
    public let total: Int
}
