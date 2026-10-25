import Foundation

/// An answer with its citation markers resolved against the numbered sources.
public struct ParsedCitations: Sendable, Equatable {
    /// The answer with invalid markers removed and valid ones renumbered in order of first use.
    public let text: String
    /// The original source numbers (1-based) in order of first use; `usedSources[i]` is now cited as `[i + 1]`.
    public let usedSources: [Int]
    /// True when markers pointing at sources that do not exist were removed.
    public let removedInvalidMarkers: Bool
}

public enum CitationParser {
    /// `[1]`, `[12]`, and the grouped forms models often write: `[1, 2]`, `[1;2]`.
    private static let markerPattern = try? NSRegularExpression(pattern: #"\[\s*(\d{1,4}(?:\s*[,;]\s*\d{1,4})*)\s*\]"#)

    /// Resolves `[n]` markers in `raw` against `sourceCount` numbered sources.
    ///
    /// A marker naming a source that does not exist is removed. Valid markers are renumbered so the
    /// first source cited is `[1]`, the next new one `[2]`, and so on, which makes the numbers
    /// shown to the user follow the order of first use.
    public static func parse(_ raw: String, sourceCount: Int) -> ParsedCitations {
        guard let regex = markerPattern else { return ParsedCitations(text: raw, usedSources: [], removedInvalidMarkers: false) }
        let range = NSRange(raw.startIndex..., in: raw)
        var order: [Int] = []
        var newNumber: [Int: Int] = [:]
        var removedInvalid = false
        var removedAny = false

        var output = ""
        var cursor = raw.startIndex
        for match in regex.matches(in: raw, range: range) {
            guard let whole = Range(match.range, in: raw), let inner = Range(match.range(at: 1), in: raw) else { continue }
            output += raw[cursor..<whole.lowerBound]
            cursor = whole.upperBound

            let numbers = raw[inner].split(whereSeparator: { $0 == "," || $0 == ";" || $0.isWhitespace }).compactMap { Int($0) }
            var rewritten = ""
            var seen = Set<Int>()
            for number in numbers where seen.insert(number).inserted {
                guard (1...max(sourceCount, 1)).contains(number), sourceCount > 0 else {
                    removedInvalid = true
                    continue
                }
                if newNumber[number] == nil {
                    order.append(number)
                    newNumber[number] = order.count
                }
                rewritten += "[\(newNumber[number] ?? number)]"
            }
            if rewritten.isEmpty { removedAny = true }
            output += rewritten
        }
        output += raw[cursor...]
        return ParsedCitations(
            text: removedAny ? tidy(output) : output, usedSources: order, removedInvalidMarkers: removedInvalid)
    }

    /// One `[n]` or grouped marker in a piece of text.
    struct Marker {
        let range: Range<String.Index>
        /// The numbers inside, in order of appearance, duplicates removed.
        let numbers: [Int]
    }

    /// Every marker in `text`, with the source numbers it names.
    static func markers(in text: String) -> [Marker] {
        guard let regex = markerPattern else { return [] }
        return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap { match in
            guard let whole = Range(match.range, in: text), let inner = Range(match.range(at: 1), in: text) else { return nil }
            var seen = Set<Int>()
            let numbers = text[inner].split(whereSeparator: { $0 == "," || $0 == ";" || $0.isWhitespace })
                .compactMap { Int($0) }.filter { seen.insert($0).inserted }
            return Marker(range: whole, numbers: numbers)
        }
    }

    /// Cleans up the gaps left by removed markers: doubled spaces and a space before punctuation.
    static func tidy(_ text: String) -> String {
        var result = text
        for (pattern, template) in [(#"[ \t]{2,}"#, " "), (#"[ \t]+([.,;:!?])"#, "$1"), (#"[ \t]+\n"#, "\n")] {
            result = result.replacingOccurrences(of: pattern, with: template, options: .regularExpression)
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
