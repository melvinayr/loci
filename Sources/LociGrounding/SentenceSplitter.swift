import Foundation
import NaturalLanguage

/// One sentence of a model's reply, located in the reply text.
struct AnswerSentence: Equatable {
    /// The sentence in the original text, trimmed, including its `[n]` markers.
    let range: Range<String.Index>
}

enum SentenceSplitter {
    /// Splits `text` into sentences, line by line so list items stay separate.
    ///
    /// Segments that carry no words of their own are folded into a neighbour so they are not scored as claims:
    /// a lone `[1]` that followed a full stop belongs to the sentence before it, and a list number such as
    /// `2.` belongs to the item after it. Segments with no letters or digits at all are dropped.
    static func split(_ text: String) -> [AnswerSentence] {
        var sentences: [AnswerSentence] = []
        text.enumerateSubstrings(in: text.startIndex..<text.endIndex, options: [.byLines, .substringNotRequired]) { _, lineRange, _, _ in
            var line: [Range<String.Index>] = []
            let tokenizer = NLTokenizer(unit: .sentence)
            tokenizer.string = text
            tokenizer.enumerateTokens(in: lineRange) { range, _ in
                if let trimmed = trim(range, in: text) { line.append(trimmed) }
                return true
            }
            sentences += fold(line, in: text).map(AnswerSentence.init)
        }
        return sentences
    }

    /// The sentence without markers, list bullets or surrounding space: what is scored as a claim.
    static func body(of range: Range<String.Index>, in text: String) -> String {
        let sentence = String(text[range])
        var cleaned = ""
        var cursor = sentence.startIndex
        for marker in CitationParser.markers(in: sentence) {
            cleaned += sentence[cursor..<marker.range.lowerBound]
            cursor = marker.range.upperBound
        }
        cleaned += sentence[cursor...]
        let withoutBullet = cleaned.replacingOccurrences(of: #"^\s*(?:[-*•]|\d{1,3}[.)])\s+"#, with: "", options: .regularExpression)
        return CitationParser.tidy(withoutBullet)
    }

    // MARK: Private

    private static func trim(_ range: Range<String.Index>, in text: String) -> Range<String.Index>? {
        var lower = range.lowerBound, upper = range.upperBound
        while lower < upper, text[lower].isWhitespace { lower = text.index(after: lower) }
        while lower < upper, text[text.index(before: upper)].isWhitespace { upper = text.index(before: upper) }
        return lower < upper ? lower..<upper : nil
    }

    private static func isBlank(_ body: String) -> Bool {
        !body.contains { $0.isLetter || $0.isNumber }
    }

    private static func isLoneListNumber(_ range: Range<String.Index>, in text: String) -> Bool {
        text[range].range(of: #"^(?:[-*•]|\d{1,3}[.)])$"#, options: .regularExpression) != nil
    }

    private static func fold(_ line: [Range<String.Index>], in text: String) -> [Range<String.Index>] {
        var result: [Range<String.Index>] = []
        var carriedStart: String.Index?  // start of list numbering waiting for its item
        for range in line {
            var range = range
            // "A fact. [1] Next fact." tokenises with the marker at the start of the next sentence; hand it back.
            if let previous = result.last, let leading = leadingMarkers(in: range, text: text) {
                result[result.count - 1] = previous.lowerBound..<leading.upperBound
                range = (trimStart(leading.upperBound, upTo: range.upperBound, in: text))..<range.upperBound
                if range.isEmpty { continue }
            }
            let start = carriedStart ?? range.lowerBound
            carriedStart = nil
            let merged = start..<range.upperBound
            if isLoneListNumber(range, in: text) {
                carriedStart = start
            } else if isBlank(body(of: merged, in: text)) {
                // Only markers or punctuation: attach to the previous sentence if there is one.
                if let previous = result.popLast() { result.append(previous.lowerBound..<range.upperBound) }
            } else {
                result.append(merged)
            }
        }
        return result
    }

    /// The run of markers a sentence opens with, if it has words after them.
    private static func leadingMarkers(in range: Range<String.Index>, text: String) -> Range<String.Index>? {
        let sentence = String(text[range])
        var end: String.Index?
        var cursor = sentence.startIndex
        for marker in CitationParser.markers(in: sentence) {
            let gap = sentence[cursor..<marker.range.lowerBound]
            guard gap.allSatisfy(\.isWhitespace) else { break }
            end = marker.range.upperBound
            cursor = marker.range.upperBound
        }
        guard let end, end < sentence.endIndex else { return nil }
        let offset = sentence.distance(from: sentence.startIndex, to: end)
        return range.lowerBound..<text.index(range.lowerBound, offsetBy: offset)
    }

    private static func trimStart(_ index: String.Index, upTo end: String.Index, in text: String) -> String.Index {
        var cursor = index
        while cursor < end, text[cursor].isWhitespace { cursor = text.index(after: cursor) }
        return cursor
    }
}
