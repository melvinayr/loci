public enum ChunkerKind: String, Sendable, Codable, CaseIterable {
    case structural, sentence, tokenWindow
}

public struct ChunkingOptions: Sendable, Equatable {
    public var kind: ChunkerKind
    public var targetTokens: Int
    public var overlapTokens: Int

    public init(kind: ChunkerKind = .structural, targetTokens: Int = 256, overlapTokens: Int = 32) {
        self.kind = kind
        self.targetTokens = max(targetTokens, 1)
        self.overlapTokens = min(max(overlapTokens, 0), max(targetTokens, 1) / 2)
    }
}

/// A passage of a document. `range` is in Unicode-scalar offsets of the normalised text
/// and `text` is exactly that slice.
public struct Chunk: Sendable, Equatable {
    public let ordinal: Int
    public let range: Range<Int>
    public let text: String
    /// First page touched; for PDFs a chunk never spans pages.
    public let page: Int?
    public let headingPath: [String]
    public let tokenCount: Int
}

/// Counts tokens in a run of scalars. The default is an estimate; the embedder's
/// tokenizer replaces it once models are loaded.
public protocol TokenCounting: Sendable {
    func count(_ scalars: ArraySlice<Unicode.Scalar>) -> Int
}

/// Roughly four scalars per token, never less than one for non-empty text.
public struct ApproximateTokenCounter: TokenCounting {
    public init() {}

    public func count(_ scalars: ArraySlice<Unicode.Scalar>) -> Int {
        scalars.isEmpty ? 0 : (scalars.count + 3) / 4
    }
}
