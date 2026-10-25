import Foundation

/// A line-based Markdown reader. It keeps headings as structure, list items and
/// table rows as their own blocks, and strips inline formatting.
enum MarkdownParser {
    static func parse(_ source: String) -> ParsedDocument {
        var builder = DocumentBuilder(kind: .markdown)
        var lines = source.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline).map(String.init)
        lines = removingFrontMatter(lines)

        var paragraph: [String] = []
        var listItem: [String]?
        var code: [String]?
        var fence: String?

        func flushParagraph() {
            if !paragraph.isEmpty { builder.add(stripInline(paragraph.joined(separator: " ")), kind: .paragraph) }
            paragraph.removeAll()
        }
        func flushListItem() {
            if let item = listItem { builder.add(stripInline(item.joined(separator: " ")), kind: .listItem) }
            listItem = nil
        }
        func flushAll() {
            flushParagraph()
            flushListItem()
        }

        for line in lines {
            if let open = fence {
                if line.trimmingCharacters(in: .whitespaces).hasPrefix(open) {
                    builder.add((code ?? []).joined(separator: " "), kind: .code)
                    code = nil
                    fence = nil
                } else {
                    code?.append(line)
                }
                continue
            }

            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty {
                flushAll()
            } else if let marker = fenceMarker(trimmed) {
                flushAll()
                fence = marker
                code = []
            } else if let (level, title) = atxHeading(trimmed) {
                flushAll()
                builder.add(stripInline(title), kind: .heading(level: level))
            } else if let level = setextLevel(trimmed), !paragraph.isEmpty {
                let title = paragraph.joined(separator: " ")
                paragraph.removeAll()
                flushListItem()
                builder.add(stripInline(title), kind: .heading(level: level))
            } else if isThematicBreak(trimmed) {
                flushAll()
            } else if let item = listItemText(line) {
                flushAll()
                listItem = [item]
            } else if trimmed.hasPrefix("|") {
                flushAll()
                if let row = tableRow(trimmed) { builder.add(stripInline(row), kind: .listItem) }
            } else if trimmed.hasPrefix(">") {
                flushListItem()
                paragraph.append(String(trimmed.drop(while: { $0 == ">" || $0 == " " })))
            } else if listItem != nil, line.hasPrefix(" ") || line.hasPrefix("\t") {
                listItem?.append(trimmed)
            } else {
                flushListItem()
                paragraph.append(trimmed)
            }
        }
        flushAll()
        if let code, !code.isEmpty { builder.add(code.joined(separator: " "), kind: .code) }

        let title = builder.firstHeading(level: 1)
        return builder.finish(title: title)
    }

    // MARK: Block recognisers

    private static func removingFrontMatter(_ lines: [String]) -> [String] {
        guard lines.first?.trimmingCharacters(in: .whitespaces) == "---" else { return lines }
        for (index, line) in lines.enumerated().dropFirst() where line.trimmingCharacters(in: .whitespaces) == "---" {
            return Array(lines[(index + 1)...])
        }
        return lines
    }

    private static func fenceMarker(_ trimmed: String) -> String? {
        for marker in ["```", "~~~"] where trimmed.hasPrefix(marker) { return marker }
        return nil
    }

    private static func atxHeading(_ trimmed: String) -> (Int, String)? {
        let hashes = trimmed.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(hashes) else { return nil }
        let rest = trimmed.dropFirst(hashes)
        guard rest.isEmpty || rest.first == " " || rest.first == "\t" else { return nil }
        var title = rest.trimmingCharacters(in: .whitespaces)
        while title.hasSuffix("#") { title.removeLast() }
        return (hashes, title.trimmingCharacters(in: .whitespaces))
    }

    private static func setextLevel(_ trimmed: String) -> Int? {
        if trimmed.allSatisfy({ $0 == "=" }) { return 1 }
        if trimmed.count >= 2, trimmed.allSatisfy({ $0 == "-" }) { return 2 }
        return nil
    }

    private static func isThematicBreak(_ trimmed: String) -> Bool {
        let compact = trimmed.filter { $0 != " " }
        guard compact.count >= 3, let first = compact.first, "-*_".contains(first) else { return false }
        return compact.allSatisfy { $0 == first }
    }

    private static let listPattern = try? NSRegularExpression(pattern: #"^\s*(?:[-*+]|\d{1,9}[.)])\s+(.*)$"#)

    private static func listItemText(_ line: String) -> String? {
        guard let match = listPattern?.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)),
              let range = Range(match.range(at: 1), in: line) else { return nil }
        return String(line[range])
    }

    private static func tableRow(_ trimmed: String) -> String? {
        let cells = trimmed.split(separator: "|", omittingEmptySubsequences: true)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        if cells.allSatisfy({ $0.allSatisfy { $0 == "-" || $0 == ":" } }) { return nil }
        return cells.joined(separator: " | ")
    }

    // MARK: Inline stripping

    private static let inlineRules: [(NSRegularExpression, String)] = {
        let rules: [(String, String)] = [
            (#"!\[([^\]]*)\]\([^)]*\)"#, "$1"),
            (#"\[([^\]]*)\]\([^)]*\)"#, "$1"),
            (#"\[([^\]]*)\]\[[^\]]*\]"#, "$1"),
            (#"<(https?://[^>\s]+)>"#, "$1"),
            (#"<[^>\n]+>"#, ""),
            (#"`+([^`]*)`+"#, "$1"),
            (#"(\*\*|__)(.+?)\1"#, "$2"),
            (#"\*(\S(?:.*?\S)?)\*"#, "$1"),
            (#"(?<![\w])_(\S(?:.*?\S)?)_(?![\w])"#, "$1"),
            (#"~~(.+?)~~"#, "$1"),
        ]
        return rules.compactMap { pattern, template in
            (try? NSRegularExpression(pattern: pattern)).map { ($0, template) }
        }
    }()

    private static let escapable = Set("\\`*_{}[]()#+-.!|>~".unicodeScalars)
    /// Escaped characters are parked in the private-use area while inline rules run, then restored.
    private static let parkingOffset: UInt32 = 0xF000

    static func stripInline(_ text: String) -> String {
        var result = ""
        var scalars = text.unicodeScalars.makeIterator()
        while let scalar = scalars.next() {
            if scalar == "\\", let next = scalars.next() {
                if escapable.contains(next), let parked = Unicode.Scalar(parkingOffset + next.value) {
                    result.unicodeScalars.append(parked)
                } else {
                    result.unicodeScalars.append(scalar)
                    result.unicodeScalars.append(next)
                }
            } else {
                result.unicodeScalars.append(scalar)
            }
        }
        for (regex, template) in inlineRules {
            result = regex.stringByReplacingMatches(
                in: result, range: NSRange(result.startIndex..., in: result), withTemplate: template)
        }
        var restored = String.UnicodeScalarView()
        for scalar in result.unicodeScalars {
            if (parkingOffset..<parkingOffset + 0x80).contains(scalar.value),
               let original = Unicode.Scalar(scalar.value - parkingOffset) {
                restored.append(original)
            } else {
                restored.append(scalar)
            }
        }
        return String(restored)
    }
}
