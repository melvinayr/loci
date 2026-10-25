/// Removes reasoning blocks (`<think>…</think>` and similar) from streamed text, including when
/// a tag is split across pieces.
///
/// Text is held back only while it could still turn out to be the start of a tag. Leading
/// whitespace at the start of the output and after a block is dropped.
struct ThinkingFilter {
    /// Open and close markers of reasoning blocks in common chat models.
    static let defaultBlocks: [(open: String, close: String)] = [
        ("<think>", "</think>"),  // Qwen3, DeepSeek-R1 and relatives
        ("<|channel>", "<channel|>"),  // Gemma 4
    ]

    private let blocks: [(open: String, close: String)]
    private var buffer = ""
    private var closing: String?  // set while inside a block
    private var dropLeadingWhitespace = true

    init(blocks: [(open: String, close: String)] = ThinkingFilter.defaultBlocks) {
        self.blocks = blocks
    }

    /// Feeds the next piece of model output and returns the visible text it releases.
    mutating func feed(_ text: String) -> String {
        buffer += text
        var output = ""
        while true {
            if let close = closing {
                guard let end = buffer.range(of: close) else {
                    buffer = Self.keepingPossiblePrefix(of: [close], in: buffer)
                    return output
                }
                buffer.removeSubrange(buffer.startIndex..<end.upperBound)
                closing = nil
                dropLeadingWhitespace = true
            } else {
                var earliest: (range: Range<String.Index>, close: String)?
                for block in blocks {
                    if let range = buffer.range(of: block.open), earliest.map({ range.lowerBound < $0.range.lowerBound }) ?? true {
                        earliest = (range, block.close)
                    }
                }
                if let found = earliest {
                    output += visible(String(buffer[..<found.range.lowerBound]))
                    buffer.removeSubrange(buffer.startIndex..<found.range.upperBound)
                    closing = found.close
                } else {
                    let held = Self.keepingPossiblePrefix(of: blocks.map(\.open), in: buffer)
                    output += visible(String(buffer.dropLast(held.count)))
                    buffer = held
                    return output
                }
            }
        }
    }

    /// Ends the stream. A block left open is discarded; a held-back partial tag was ordinary text.
    mutating func finish() -> String {
        defer { buffer = ""; closing = nil }
        return closing == nil ? visible(buffer) : ""
    }

    private mutating func visible(_ text: String) -> String {
        guard dropLeadingWhitespace else { return text }
        let trimmed = text.drop(while: \.isWhitespace)
        if !trimmed.isEmpty { dropLeadingWhitespace = false }
        return String(trimmed)
    }

    /// The longest suffix of `text` that is a proper prefix of one of `tags`.
    private static func keepingPossiblePrefix(of tags: [String], in text: String) -> String {
        var best = 0
        for tag in tags {
            let limit = min(tag.count - 1, text.count)
            guard limit > 0 else { continue }
            for length in stride(from: limit, to: best, by: -1) where tag.hasPrefix(text.suffix(length)) {
                best = length
                break
            }
        }
        return String(text.suffix(best))
    }
}
