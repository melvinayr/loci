import Foundation
import Testing
@testable import LociIngest

@Suite("Normalizer")
struct NormalizerTests {
    @Test func composesCollapsesAndTrims() {
        #expect(Normalizer.normalize("  cafe\u{301}\n\t and \u{00A0} tea  ") == "caf\u{E9} and tea")
    }

    @Test func stripsInvisibleAndControlScalars() {
        #expect(Normalizer.normalize("a\u{200B}b\u{00AD}c\u{FEFF}d\u{0007}e") == "abcde")
    }

    @Test(arguments: ["  already clean ", "e\u{301}\u{301} x", "a\r\nb", "\u{200B}", ""])
    func isIdempotent(_ input: String) {
        let once = Normalizer.normalize(input)
        #expect(Normalizer.normalize(once) == once)
    }
}

@Suite("Markdown parser")
struct MarkdownParserTests {
    let document = try! DocumentParser.parse(fileAt: Fixtures.url("sample.md"))

    @Test func titleComesFromFirstH1() {
        #expect(document.title == "Field Notes")
        #expect(document.kind == .markdown)
    }

    @Test func stripsFormattingAndFrontMatter() {
        let text = document.text
        #expect(!text.contains("**") && !text.contains("](") && !text.contains("`"))
        #expect(!text.contains("front matter") && !text.contains("tags:"))
        #expect(text.contains("Intro paragraph with bold, italic, code and a link plus snake_case_name."))
        #expect(text.contains("a *literal* star"))
    }

    @Test func keepsStructureAndHeadingPaths() throws {
        let item = try #require(document.blocks.first { document.text.scalarSlice($0.range) == "First item about sampling" })
        #expect(item.kind == .listItem)
        #expect(item.headingPath == ["Field Notes", "Sampling"])

        let continued = document.blocks.first { document.text.scalarSlice($0.range) == "Second item continues here" }
        #expect(continued != nil)

        let code = document.blocks.first { document.text.scalarSlice($0.range) == "let sampleSize = 412" }
        #expect(code?.kind == .code)
    }

    @Test func tablesBecomeRowsWithoutSeparators() {
        let texts = document.blocks.map { document.text.scalarSlice($0.range) }
        #expect(texts.contains("Name | Value"))
        #expect(texts.contains("alpha | 1"))
        #expect(!texts.contains { $0.contains("---") })
    }

    @Test func setextHeadingsAndNFC() {
        #expect(document.blocks.contains { $0.kind == .heading(level: 1) && document.text.scalarSlice($0.range) == "Setext Heading" })
        #expect(document.text.contains("Caf\u{E9} ☕ notes"))
        #expect(!document.text.unicodeScalars.contains("\u{301}"))
    }

    @Test func blocksAreOrderedNonOverlappingAndSliceCleanly() {
        var previousEnd = 0
        for block in document.blocks {
            #expect(block.range.lowerBound >= previousEnd)
            #expect(!document.text.scalarSlice(block.range).isEmpty)
            previousEnd = block.range.upperBound
        }
        #expect(previousEnd <= document.text.scalarCount)
    }
}

@Suite("HTML parser")
struct HTMLParserTests {
    let document = try! DocumentParser.parse(fileAt: Fixtures.url("sample.html"))

    @Test func titleAndEntities() {
        #expect(document.title == "Trial & Results")
        #expect(document.text.contains("The trial enrolled 412 participants in total — all randomised & followed up."))
    }

    @Test func dropsChromeScriptsStylesAndComments() {
        for junk in ["Home", "About", "Copyright", "color", "scripted", "hidden"] {
            #expect(!document.text.contains(junk), "found “\(junk)”")
        }
    }

    @Test func keepsHeadingsListsAndPreformatted() {
        let kinds = document.blocks.map(\.kind)
        #expect(kinds.contains(.heading(level: 1)) && kinds.contains(.heading(level: 2)))
        #expect(kinds.filter { $0 == .listItem }.count == 2)
        #expect(kinds.contains(.code))
        let item = document.blocks.first { $0.kind == .listItem }
        #expect(item?.headingPath == ["Trial Overview", "Methods"])
    }

    @Test func malformedMarkupDoesNotCrash() {
        for broken in ["<p>unclosed", "<<<>>>", "<a href=\"x", "&#xZZ; &amp", "<script>never closed", ""] {
            _ = DocumentParser.parse(text: broken, kind: .html)
        }
    }
}

@Suite("Plain text parser")
struct PlainTextParserTests {
    @Test func splitsParagraphsOnBlankLinesAndHandlesCRLF() throws {
        let document = try DocumentParser.parse(fileAt: Fixtures.url("sample.txt"))
        let texts = document.blocks.map { document.text.scalarSlice($0.range) }
        #expect(texts == [
            "Plain text fixture. Second line of the same paragraph.",
            "Second paragraph with emoji 😀 and caf\u{E9}.",
            "Third paragraph.",
        ])
        #expect(document.text.contains("\r") == false)
    }

    @Test func rejectsUnsupportedAndEmptyFiles() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let unsupported = directory.appendingPathComponent("a.docx")
        try Data([1, 2, 3]).write(to: unsupported)
        #expect(throws: IngestError.self) { try DocumentParser.parse(fileAt: unsupported) }

        let empty = directory.appendingPathComponent("empty.txt")
        try "  \n\n ".write(to: empty, atomically: true, encoding: .utf8)
        #expect(throws: IngestError.self) { try DocumentParser.parse(fileAt: empty) }
    }

    @Test func decodesUTF16AndLatin1() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let utf16 = directory.appendingPathComponent("u16.txt")
        try "héllo wörld".data(using: .utf16)!.write(to: utf16)
        #expect(try DocumentParser.parse(fileAt: utf16).text == "héllo wörld")

        let latin1 = directory.appendingPathComponent("l1.txt")
        try "caf\u{E9}".data(using: .isoLatin1)!.write(to: latin1)
        #expect(try DocumentParser.parse(fileAt: latin1).text == "caf\u{E9}")
    }
}

@Suite("PDF parser")
struct PDFParserTests {
    @Test func keepsPageMapAndPageNumbers() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("three.pdf")
        try writePDF(pages: ["Alpha page one text.", nil, "Gamma page three text."], to: url)

        let document = try DocumentParser.parse(fileAt: url)
        #expect(document.kind == .pdf)
        #expect(document.pageCount == 3)
        #expect(document.pagesWithoutText == 1)

        let ranges = try #require(document.pageRanges)
        #expect(ranges.count == 3)
        #expect(ranges[1] == nil)
        let first = try #require(ranges[0]), third = try #require(ranges[2])
        #expect(document.text.scalarSlice(first).contains("Alpha"))
        #expect(document.text.scalarSlice(third).contains("Gamma"))
        #expect(first.upperBound <= third.lowerBound)
        #expect(document.blocks.map(\.page).contains(1) && document.blocks.map(\.page).contains(3))
    }

    @Test func scannedPDFIsUnreadable() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("scan.pdf")
        try writePDF(pages: [nil, nil], to: url)
        #expect(throws: IngestError.unreadable(reason: "the PDF has no text layer (scanned PDFs need OCR, which arrives in a later release)")) {
            try DocumentParser.parse(fileAt: url)
        }
    }

    @Test func encryptedPDFIsUnreadable() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("locked.pdf")
        try writePDF(pages: ["secret"], to: url, password: "hunter2")
        #expect(throws: IngestError.self) { try DocumentParser.parse(fileAt: url) }
    }

    @Test func nonPDFBytesAreUnreadable() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("fake.pdf")
        try "not a pdf".write(to: url, atomically: true, encoding: .utf8)
        #expect(throws: IngestError.self) { try DocumentParser.parse(fileAt: url) }
    }
}
