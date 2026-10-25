import Foundation
import PDFKit

/// Locates a citation's passage in a PDF so a viewer can scroll to it and mark it.
public enum PDFHighlighter {
    public struct Target {
        public let page: PDFPage
        /// The passage on that page; nil if it could not be found there (the page is still the right place to look).
        public let selection: PDFSelection?
    }

    /// The page cited (1-based, as in `Citation.page`) and the passage's selection on it.
    public static func target(in document: PDFDocument, page pageNumber: Int, excerpt: String) -> Target? {
        guard pageNumber >= 1, pageNumber <= document.pageCount, let page = document.page(at: pageNumber - 1) else { return nil }
        guard let text = page.string, let range = PassageLocator.locate(excerpt: excerpt, inPageText: text),
              let selection = page.selection(for: range) else {
            return Target(page: page, selection: nil)
        }
        selection.color = .yellow
        return Target(page: page, selection: selection)
    }
}
