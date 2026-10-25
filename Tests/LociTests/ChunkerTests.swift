import Foundation
import Testing
@testable import LociIngest

/// Invariants that must hold for every parser × chunker combination.
@Suite("Chunker invariants")
struct ChunkerInvariantTests {
    static func documents() throws -> [(name: String, document: ParsedDocument)] {
        var result = try ["sample.md", "sample.html", "sample.txt", "long.md"].map {
            (name: $0, document: try DocumentParser.parse(fileAt: Fixtures.url($0)))
        }
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("long.pdf")
        let page = (1...40).map { "Sentence \($0) of the page talks about subject \($0 % 5) at some length." }
            .joined(separator: " ")
        try writePDF(pages: [page, nil, page + "\n\nA second paragraph on page four.", "Short last page."], to: url)
        result.append((name: "long.pdf", document: try DocumentParser.parse(fileAt: url)))
        return result
    }

    static let optionSets: [ChunkingOptions] = ChunkerKind.allCases.flatMap { kind in
        [ChunkingOptions(kind: kind, targetTokens: 40, overlapTokens: 8),
         ChunkingOptions(kind: kind, targetTokens: 256, overlapTokens: 32),
         ChunkingOptions(kind: kind, targetTokens: 5, overlapTokens: 0)]
    }

    @Test(arguments: optionSets)
    func everyChunkSlicesToItsRangeAndStaysOrdered(options: ChunkingOptions) throws {
        for (name, document) in try Self.documents() {
            let chunks = Chunker.chunk(document, options: options)
            #expect(!chunks.isEmpty, "\(name) produced no chunks")
            var previousStart = -1
            for (index, chunk) in chunks.enumerated() {
                #expect(chunk.ordinal == index)
                #expect(chunk.range.lowerBound >= previousStart, "\(name) chunk \(index) out of order")
                #expect(chunk.range.upperBound <= document.text.scalarCount)
                #expect(chunk.text == document.text.scalarSlice(chunk.range), "\(name) chunk \(index) text ≠ slice")
                #expect(!chunk.text.isEmpty && chunk.text.first?.isWhitespace == false && chunk.text.last?.isWhitespace == false)
                #expect(chunk.tokenCount > 0)
                previousStart = chunk.range.lowerBound
            }
        }
    }

    @Test(arguments: optionSets)
    func chunksCoverEveryNonWhitespaceScalar(options: ChunkingOptions) throws {
        for (name, document) in try Self.documents() {
            let scalars = Array(document.text.unicodeScalars)
            var covered = [Bool](repeating: false, count: scalars.count)
            for chunk in Chunker.chunk(document, options: options) {
                for i in chunk.range { covered[i] = true }
            }
            for (i, scalar) in scalars.enumerated() where !scalar.properties.isWhitespace {
                #expect(covered[i], "\(name): scalar \(i) is in no chunk")
            }
        }
    }

    @Test(arguments: optionSets)
    func pdfChunksNeverSpanPagesAndPageContainsRange(options: ChunkingOptions) throws {
        let (_, document) = try #require(try Self.documents().first { $0.name == "long.pdf" })
        let pageRanges = try #require(document.pageRanges)
        for chunk in Chunker.chunk(document, options: options) {
            let page = try #require(chunk.page)
            let range = try #require(pageRanges[page - 1])
            #expect(range.contains(chunk.range.lowerBound) && chunk.range.upperBound <= range.upperBound,
                    "chunk \(chunk.ordinal) escapes page \(page)")
        }
    }

    @Test func chunksRespectTheTargetUnlessASingleUnitCannotShrink() throws {
        let document = try DocumentParser.parse(fileAt: Fixtures.url("long.md"))
        for kind in ChunkerKind.allCases {
            let chunks = Chunker.chunk(document, options: ChunkingOptions(kind: kind, targetTokens: 40, overlapTokens: 8))
            // Joining separators ("\n\n") can add a token or two beyond the unit sum.
            #expect(chunks.allSatisfy { $0.tokenCount <= 40 + 2 }, "\(kind) exceeded target: \(chunks.map(\.tokenCount))")
        }
    }
}

@Suite("Chunker behaviour")
struct ChunkerBehaviourTests {
    @Test func structuralChunksStartAtHeadingsWithHeadingPaths() throws {
        let document = try DocumentParser.parse(fileAt: Fixtures.url("long.md"))
        let chunks = Chunker.chunk(document, options: ChunkingOptions(kind: .structural, targetTokens: 60, overlapTokens: 0))

        let partTwo = try #require(chunks.first { $0.text.hasPrefix("Part Two") })
        #expect(partTwo.headingPath == ["Long Document", "Part Two"])
        let short = try #require(chunks.last)
        #expect(short.text.hasPrefix("Short"))
        #expect(short.headingPath == ["Long Document", "Short"])
        // A section never bleeds into the previous one.
        #expect(!chunks.contains { $0.text.contains("Part Two") && $0.text.contains("Sentence number 1 ") })
    }

    @Test func overlapRepeatsTrailingUnits() throws {
        let document = DocumentParser.parse(
            text: (1...30).map { "Sentence \($0) is here." }.joined(separator: " "), kind: .text)
        let chunks = Chunker.chunk(document, options: ChunkingOptions(kind: .sentence, targetTokens: 20, overlapTokens: 8))
        #expect(chunks.count > 2)
        for (a, b) in zip(chunks, chunks.dropFirst()) {
            #expect(b.range.lowerBound < a.range.upperBound, "chunks \(a.ordinal) and \(b.ordinal) do not overlap")
            #expect(b.range.upperBound > a.range.upperBound, "chunk \(b.ordinal) adds nothing")
        }
    }

    @Test func carriedOverlapIsAlwaysContiguous() throws {
        // Regression: a small early unit must not be carried across a larger one.
        let document = try DocumentParser.parse(fileAt: Fixtures.url("long.md"))
        let chunks = Chunker.chunk(document, options: ChunkingOptions(kind: .sentence, targetTokens: 40, overlapTokens: 8))
        for (a, b) in zip(chunks, chunks.dropFirst()) {
            #expect(b.range.lowerBound >= a.range.lowerBound)
            #expect(b.range.lowerBound >= a.range.upperBound - 60, "chunk \(b.ordinal) starts far before the previous chunk's end")
        }
    }

    @Test func zeroOverlapProducesDisjointChunks() throws {
        let document = DocumentParser.parse(
            text: (1...30).map { "Sentence \($0) is here." }.joined(separator: " "), kind: .text)
        let chunks = Chunker.chunk(document, options: ChunkingOptions(kind: .sentence, targetTokens: 20, overlapTokens: 0))
        for (a, b) in zip(chunks, chunks.dropFirst()) {
            #expect(b.range.lowerBound >= a.range.upperBound)
        }
    }

    @Test func oversizedBlocksSplitAtSentencesNotMidSentence() {
        let sentence = "This is one complete sentence of moderate length."
        let document = DocumentParser.parse(text: Array(repeating: sentence, count: 20).joined(separator: " "), kind: .text)
        let chunks = Chunker.chunk(document, options: ChunkingOptions(kind: .structural, targetTokens: 30, overlapTokens: 0))
        #expect(chunks.count > 1)
        #expect(chunks.allSatisfy { $0.text.hasPrefix("This") && $0.text.hasSuffix(".") })
    }

    @Test func aSingleHugeWordIsHardSplitAndTerminates() {
        let document = DocumentParser.parse(text: String(repeating: "x", count: 5_000), kind: .text)
        let chunks = Chunker.chunk(document, options: ChunkingOptions(kind: .tokenWindow, targetTokens: 50, overlapTokens: 10))
        #expect(chunks.count > 1)
        #expect(chunks.map(\.text.count).reduce(0, +) >= 5_000)
    }

    @Test func emptyDocumentHasNoChunks() {
        #expect(Chunker.chunk(DocumentParser.parse(text: "  \n ", kind: .text)).isEmpty)
    }

    @Test func optionsAreClamped() {
        let options = ChunkingOptions(kind: .structural, targetTokens: 0, overlapTokens: 999)
        #expect(options.targetTokens == 1 && options.overlapTokens == 0)
    }

    @Test func unicodeOffsetsAreScalarBased() throws {
        let document = DocumentParser.parse(text: "😀😀 first sentence. café second sentence.", kind: .text)
        let chunks = Chunker.chunk(document, options: ChunkingOptions(kind: .sentence, targetTokens: 6, overlapTokens: 0))
        #expect(chunks.count >= 2)
        for chunk in chunks { #expect(chunk.text == document.text.scalarSlice(chunk.range)) }
    }
}
