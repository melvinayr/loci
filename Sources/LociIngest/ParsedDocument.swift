/// The format a document was parsed from.
public enum SourceKind: String, Sendable, Codable, CaseIterable {
    case pdf, markdown, html, text
}

public enum BlockKind: Sendable, Equatable {
    case heading(level: Int)
    case paragraph
    case listItem
    case code
}

/// A structural unit of a document. `range` is in Unicode-scalar offsets of
/// `ParsedDocument.text`.
public struct ParsedBlock: Sendable, Equatable {
    public let range: Range<Int>
    public let kind: BlockKind
    /// 1-based page, PDFs only.
    public let page: Int?
    /// Enclosing headings, outermost first; includes the block itself for headings.
    public let headingPath: [String]
}

/// Normalised text plus the structure needed to chunk it with exact offsets.
///
/// `text` is the normalised blocks joined by a blank line (`"\n\n"`).
public struct ParsedDocument: Sendable {
    public let kind: SourceKind
    public let title: String?
    public let text: String
    public let blocks: [ParsedBlock]
    public let pageCount: Int?
    /// PDFs only: for each page, the range of text it contributed (nil when the page has no text).
    public let pageRanges: [Range<Int>?]?
    /// PDF pages that had no text layer.
    public let pagesWithoutText: Int
}

/// Accumulates raw blocks, normalising each and recording exact offsets as it goes.
struct DocumentBuilder {
    let kind: SourceKind
    private var scalars: [Unicode.Scalar] = []
    private var blocks: [ParsedBlock] = []
    private var headings: [(level: Int, title: String)] = []
    private var pageExtents: [Int: Range<Int>] = [:]

    init(kind: SourceKind) {
        self.kind = kind
    }

    var isEmpty: Bool { blocks.isEmpty }
    var blockCount: Int { blocks.count }
    /// Text of the first heading at `level`.
    func firstHeading(level: Int) -> String? {
        for block in blocks {
            if case .heading(let found) = block.kind, found == level { return block.headingPath.last }
        }
        return nil
    }

    mutating func add(_ raw: String, kind: BlockKind, page: Int? = nil) {
        let text = Normalizer.normalize(raw)
        guard !text.isEmpty else { return }

        if !scalars.isEmpty { scalars.append(contentsOf: ["\n", "\n"]) }
        let start = scalars.count
        scalars.append(contentsOf: text.unicodeScalars)
        let range = start..<scalars.count

        if case .heading(let level) = kind {
            while let last = headings.last, last.level >= level { headings.removeLast() }
            headings.append((level, text))
        }
        blocks.append(ParsedBlock(range: range, kind: kind, page: page, headingPath: headings.map(\.title)))

        if let page {
            let merged = pageExtents[page].map { $0.lowerBound..<range.upperBound } ?? range
            pageExtents[page] = merged
        }
    }

    func finish(title: String?, pageCount: Int? = nil, pagesWithoutText: Int = 0) -> ParsedDocument {
        var pageRanges: [Range<Int>?]?
        if let pageCount {
            pageRanges = (1...max(pageCount, 1)).map { pageExtents[$0] }
            if pageCount == 0 { pageRanges = [] }
        }
        return ParsedDocument(
            kind: kind,
            title: title.map(Normalizer.normalize).flatMap { $0.isEmpty ? nil : $0 },
            text: String(String.UnicodeScalarView(scalars)),
            blocks: blocks,
            pageCount: pageCount,
            pageRanges: pageRanges,
            pagesWithoutText: pagesWithoutText
        )
    }
}
