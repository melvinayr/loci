import Foundation
import PDFKit

enum PDFParser {
    /// Reads a PDF page by page, keeping a page → offset map. Pages without a text
    /// layer are counted; a document with no text at all is unreadable.
    static func parse(fileAt url: URL) throws -> ParsedDocument {
        guard let document = PDFDocument(url: url) else {
            throw IngestError.unreadable(reason: "not a readable PDF")
        }
        if document.isEncrypted && document.isLocked {
            throw IngestError.unreadable(reason: "the PDF is password protected")
        }

        var builder = DocumentBuilder(kind: .pdf)
        var withoutText = 0
        for pageIndex in 0..<document.pageCount {
            let beforeCount = builder.blockCount
            if let text = document.page(at: pageIndex)?.string {
                for paragraph in paragraphs(of: text) {
                    builder.add(paragraph, kind: .paragraph, page: pageIndex + 1)
                }
            }
            if builder.blockCount == beforeCount { withoutText += 1 }
        }

        guard !builder.isEmpty else {
            throw IngestError.unreadable(
                reason: "the PDF has no text layer (scanned PDFs need OCR, which arrives in a later release)")
        }

        let title = document.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String
        return builder.finish(title: title, pageCount: document.pageCount, pagesWithoutText: withoutText)
    }

    /// Splits page text on blank lines.
    private static func paragraphs(of text: String) -> [String] {
        var result: [String] = []
        var current: [Substring] = []
        for line in text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline) {
            if line.allSatisfy(\.isWhitespace) {
                if !current.isEmpty { result.append(current.joined(separator: " ")) }
                current.removeAll()
            } else {
                current.append(line)
            }
        }
        if !current.isEmpty { result.append(current.joined(separator: " ")) }
        return result
    }
}
