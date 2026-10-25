import Foundation

/// A small tolerant HTML reader. Drops scripts, styles, navigation and other chrome,
/// and keeps headings, paragraphs, list items and preformatted text as blocks.
enum HTMLParser {
    private static let skipped: Set<String> = [
        "script", "style", "noscript", "template", "svg", "nav", "footer", "aside", "iframe",
    ]
    private static let rawText: Set<String> = ["script", "style"]
    private static let voidElements: Set<String> = [
        "br", "hr", "img", "meta", "link", "input", "area", "base", "col", "embed", "source", "track", "wbr",
    ]
    private static let blockElements: Set<String> = [
        "p", "div", "section", "article", "main", "header", "br", "hr", "tr", "td", "th", "table", "ul", "ol",
        "dl", "dt", "dd", "blockquote", "figure", "figcaption", "form", "fieldset", "details", "summary",
        "address", "caption",
    ]

    static func parse(_ html: String) -> ParsedDocument {
        let chars = Array(html.unicodeScalars)
        var builder = DocumentBuilder(kind: .html)
        var buffer = String.UnicodeScalarView()
        var kind: BlockKind = .paragraph
        var title = String.UnicodeScalarView()
        var inTitle = false
        var skipName: String?
        var skipDepth = 0
        var index = 0

        func flush() {
            builder.add(decodeEntities(String(buffer)), kind: kind)
            buffer.removeAll()
        }

        while index < chars.count {
            let scalar = chars[index]
            guard scalar == "<" else {
                if skipName == nil {
                    if inTitle { title.append(scalar) } else { buffer.append(scalar) }
                }
                index += 1
                continue
            }

            if matches("<!--", chars, at: index) {
                index = find("-->", chars, from: index + 4).map { $0 + 3 } ?? chars.count
                continue
            }
            guard let tag = readTag(chars, at: index) else {
                if skipName == nil { buffer.append(scalar) }
                index += 1
                continue
            }
            index = tag.end

            if let name = skipName {
                if tag.name == name {
                    if tag.closing { skipDepth -= 1 } else if !tag.selfClosing { skipDepth += 1 }
                    if skipDepth == 0 { skipName = nil }
                }
                continue
            }

            if tag.name == "title" {
                if tag.closing {
                    inTitle = false
                } else {
                    inTitle = true
                }
                continue
            }

            if skipped.contains(tag.name), !tag.closing, !tag.selfClosing, !voidElements.contains(tag.name) {
                if rawText.contains(tag.name) {
                    index = find("</\(tag.name)", chars, from: index, caseInsensitive: true)
                        .flatMap { find(">", chars, from: $0) }.map { $0 + 1 } ?? chars.count
                } else {
                    skipName = tag.name
                    skipDepth = 1
                }
                continue
            }

            if let level = headingLevel(tag.name) {
                flush()
                kind = tag.closing ? .paragraph : .heading(level: level)
            } else if tag.name == "li" {
                flush()
                kind = tag.closing ? .paragraph : .listItem
            } else if tag.name == "pre" {
                flush()
                kind = tag.closing ? .paragraph : .code
            } else if blockElements.contains(tag.name) {
                flush()
            }
        }
        flush()

        let documentTitle = Normalizer.normalize(decodeEntities(String(title)))
        return builder.finish(title: documentTitle.isEmpty ? nil : documentTitle)
    }

    // MARK: Tokenising

    private struct Tag {
        let name: String
        let closing: Bool
        let selfClosing: Bool
        let end: Int
    }

    private static func readTag(_ chars: [Unicode.Scalar], at start: Int) -> Tag? {
        var i = start + 1
        var closing = false
        if i < chars.count, chars[i] == "/" { closing = true; i += 1 }
        var name = ""
        while i < chars.count, chars[i].properties.isAlphabetic || chars[i].properties.numericType != nil {
            name.unicodeScalars.append(chars[i])
            i += 1
        }
        guard !name.isEmpty else { return nil }

        var quote: Unicode.Scalar?
        var lastNonSpace: Unicode.Scalar = " "
        while i < chars.count {
            let c = chars[i]
            if let q = quote {
                if c == q { quote = nil }
            } else if c == "\"" || c == "'" {
                quote = c
            } else if c == ">" {
                return Tag(name: name.lowercased(), closing: closing, selfClosing: lastNonSpace == "/", end: i + 1)
            }
            if !c.properties.isWhitespace { lastNonSpace = c }
            i += 1
        }
        return nil
    }

    private static func headingLevel(_ name: String) -> Int? {
        guard name.count == 2, name.first == "h", let level = Int(name.dropFirst()), (1...6).contains(level) else {
            return nil
        }
        return level
    }

    private static func matches(_ pattern: String, _ chars: [Unicode.Scalar], at index: Int) -> Bool {
        let p = Array(pattern.unicodeScalars)
        guard index + p.count <= chars.count else { return false }
        return chars[index..<index + p.count].elementsEqual(p)
    }

    private static func find(_ pattern: String, _ chars: [Unicode.Scalar], from: Int, caseInsensitive: Bool = false) -> Int? {
        let p = Array((caseInsensitive ? pattern.lowercased() : pattern).unicodeScalars)
        guard !p.isEmpty, chars.count >= p.count, from <= chars.count - p.count else { return nil }
        for i in from...(chars.count - p.count) {
            var hit = true
            for j in 0..<p.count {
                let c = caseInsensitive ? Unicode.Scalar(String(chars[i + j]).lowercased().unicodeScalars.first ?? chars[i + j]) : chars[i + j]
                if c != p[j] { hit = false; break }
            }
            if hit { return i }
        }
        return nil
    }

    // MARK: Entities

    private static let namedEntities: [String: String] = [
        "amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": " ", "ndash": "–", "mdash": "—",
        "hellip": "…", "lsquo": "‘", "rsquo": "’", "ldquo": "“", "rdquo": "”", "copy": "©", "reg": "®",
        "trade": "™", "deg": "°", "middot": "·", "bull": "•", "laquo": "«", "raquo": "»", "euro": "€",
        "pound": "£", "times": "×", "eacute": "é", "egrave": "è", "agrave": "à", "uuml": "ü", "ouml": "ö",
        "auml": "ä", "ccedil": "ç", "ntilde": "ñ",
    ]

    static func decodeEntities(_ text: String) -> String {
        guard text.contains("&") else { return text }
        var output = String.UnicodeScalarView()
        let scalars = Array(text.unicodeScalars)
        var i = 0
        while i < scalars.count {
            guard scalars[i] == "&", let semicolon = scalars[i...].prefix(12).firstIndex(of: ";") else {
                output.append(scalars[i])
                i += 1
                continue
            }
            let body = String(String.UnicodeScalarView(scalars[(i + 1)..<semicolon]))
            if let decoded = decode(entity: body) {
                output.append(contentsOf: decoded.unicodeScalars)
                i = semicolon + 1
            } else {
                output.append(scalars[i])
                i += 1
            }
        }
        return String(output)
    }

    private static func decode(entity body: String) -> String? {
        if body.hasPrefix("#") {
            let digits = body.dropFirst()
            let value = digits.first == "x" || digits.first == "X"
                ? UInt32(digits.dropFirst(), radix: 16) : UInt32(digits, radix: 10)
            return value.flatMap(Unicode.Scalar.init).map { String(Character($0)) }
        }
        return namedEntities[body]
    }
}
