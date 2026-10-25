enum PlainTextParser {
    /// Paragraphs are separated by blank lines.
    static func parse(_ text: String) -> ParsedDocument {
        var builder = DocumentBuilder(kind: .text)
        var paragraph: [Substring] = []
        func flush() {
            builder.add(paragraph.joined(separator: " "), kind: .paragraph)
            paragraph.removeAll()
        }
        for line in text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline) {
            if line.allSatisfy(\.isWhitespace) {
                flush()
            } else {
                paragraph.append(line)
            }
        }
        flush()
        return builder.finish(title: nil)
    }
}
