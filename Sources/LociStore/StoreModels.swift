import Foundation

/// A document as persisted (without its text; see `Store.documentText`).
public struct StoredDocument: Sendable, Hashable {
    public let id: String
    public let title: String
    public let uri: String?
    public let kind: String
    public let pageCount: Int?
    public let chunkCount: Int
    public let contentHash: String
    public let metadata: [String: String]
    public let addedAt: Date
}

public struct StoredChunk: Sendable, Hashable {
    public let id: Int64
    public let documentID: String
    public let ordinal: Int
    public let page: Int?
    public let start: Int
    public let end: Int
    public let headingPath: [String]
    public let tokenCount: Int
    public let text: String
}

/// A document to insert. `text` is the normalised text that chunk offsets refer to.
public struct NewDocument: Sendable {
    public var title: String
    public var uri: String?
    public var kind: String
    public var pageCount: Int?
    public var contentHash: String
    public var metadata: [String: String]
    public var text: String
    public var pageRanges: [Range<Int>?]?

    public init(
        title: String, uri: String?, kind: String, pageCount: Int?, contentHash: String,
        metadata: [String: String], text: String, pageRanges: [Range<Int>?]?
    ) {
        self.title = title
        self.uri = uri
        self.kind = kind
        self.pageCount = pageCount
        self.contentHash = contentHash
        self.metadata = metadata
        self.text = text
        self.pageRanges = pageRanges
    }
}

public struct NewChunk: Sendable {
    public var ordinal: Int
    public var page: Int?
    public var start: Int
    public var end: Int
    public var headingPath: [String]
    public var tokenCount: Int
    public var text: String

    public init(ordinal: Int, page: Int?, start: Int, end: Int, headingPath: [String], tokenCount: Int, text: String) {
        self.ordinal = ordinal
        self.page = page
        self.start = start
        self.end = end
        self.headingPath = headingPath
        self.tokenCount = tokenCount
        self.text = text
    }
}

public enum InsertOutcome: Sendable {
    /// The document was stored.
    case inserted(StoredDocument)
    /// A document with identical normalised text already exists; nothing was written.
    case existing(StoredDocument)
}

public struct KeywordHit: Sendable, Hashable {
    public let chunk: StoredChunk
    /// BM25 relevance, higher is better.
    public let score: Double
}

public struct StoreStats: Sendable, Equatable {
    public let documentCount: Int
    public let chunkCount: Int
    public let vectorCount: Int
    public let databaseBytes: UInt64
}

/// The embedding model a store's vectors were built with.
public struct StoreEmbedder: Sendable, Equatable {
    public let id: String
    public let dimension: Int

    public init(id: String, dimension: Int) {
        self.id = id
        self.dimension = dimension
    }
}

public struct VectorHit: Sendable, Hashable {
    public let chunkID: Int64
    /// Cosine similarity, higher is better.
    public let score: Float
}
