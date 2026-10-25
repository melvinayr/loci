import Foundation

/// Entry point for turning files or strings into `ParsedDocument`s.
public enum DocumentParser {
    public static let supportedExtensions: Set<String> = ["pdf", "md", "markdown", "html", "htm", "txt"]

    public static func kind(forExtension ext: String) -> SourceKind? {
        switch ext.lowercased() {
        case "pdf": .pdf
        case "md", "markdown": .markdown
        case "html", "htm": .html
        case "txt": .text
        default: nil
        }
    }

    /// Parses a file. Throws `IngestError` for unsupported or unreadable files.
    public static func parse(fileAt url: URL) throws -> ParsedDocument {
        guard let kind = kind(forExtension: url.pathExtension) else {
            throw IngestError.unsupported(reason: "unsupported file type “\(url.pathExtension)”")
        }
        let parsed: ParsedDocument
        if kind == .pdf {
            parsed = try PDFParser.parse(fileAt: url)
        } else {
            let data: Data
            do {
                data = try Data(contentsOf: url)
            } catch {
                throw IngestError.unreadable(reason: "the file could not be read")
            }
            parsed = parse(text: decode(data), kind: kind)
        }
        guard !parsed.text.isEmpty else {
            throw IngestError.unreadable(reason: "the document contains no text")
        }
        return parsed
    }

    /// Parses in-memory text as `kind` (`.pdf` is not supported here and is treated as plain text).
    public static func parse(text: String, kind: SourceKind = .text) -> ParsedDocument {
        switch kind {
        case .markdown: MarkdownParser.parse(text)
        case .html: HTMLParser.parse(text)
        case .text, .pdf: PlainTextParser.parse(text)
        }
    }

    /// UTF-8 (with or without BOM), then UTF-16 with a BOM, then Latin-1 as a last resort.
    static func decode(_ data: Data) -> String {
        if data.starts(with: [0xFF, 0xFE]) || data.starts(with: [0xFE, 0xFF]),
           let text = String(data: data, encoding: .utf16) {
            return text
        }
        if let text = String(data: data, encoding: .utf8) { return text }
        return String(data: data, encoding: .isoLatin1) ?? ""
    }
}
