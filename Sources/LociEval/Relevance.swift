import Foundation
import Loci

/// Decides whether retrieved passages are the ground-truth ones.
enum Relevance {
    /// For each result, whether it matches a relevant passage that no earlier result already matched.
    static func flags(for results: [SearchResult], relevant: [RelevantPassage]) -> [Bool] {
        var unmatched = Array(relevant.indices)
        return results.map { result in
            guard let position = unmatched.firstIndex(where: { matches(result, relevant[$0]) }) else { return false }
            unmatched.remove(at: position)
            return true
        }
    }

    static func matches(_ result: SearchResult, _ passage: RelevantPassage) -> Bool {
        matches(document: result.document, page: result.page, text: result.text, passage)
    }

    /// Whether a retrieved or cited passage (document, page and text) is the ground-truth `passage`.
    static func matches(document: Document, page: Int?, text: String, _ passage: RelevantPassage) -> Bool {
        guard documentName(document) == passage.document else { return false }
        if let wanted = passage.page, page != wanted { return false }
        if let snippet = passage.snippet {
            return squash(text).contains(squash(snippet))
        }
        return true
    }

    /// Letters, digits and single spaces only, lowercased: for comparing an answer with a reference answer.
    static func normalize(_ text: String) -> String {
        String(text.lowercased().map { $0.isLetter || $0.isNumber ? $0 : " " })
            .split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    /// The file name a document was added from, falling back to its title.
    static func documentName(_ document: Document) -> String {
        if let uri = document.uri, let url = URL(string: uri), url.isFileURL { return url.lastPathComponent }
        return document.title
    }

    /// Lowercases and collapses whitespace so formatting differences do not matter.
    static func squash(_ text: String) -> String {
        text.lowercased().split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}
