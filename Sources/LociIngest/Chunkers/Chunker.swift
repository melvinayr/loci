import Foundation
import NaturalLanguage

/// Splits a `ParsedDocument` into chunks with exact offsets.
///
/// All three strategies share one packer. They differ only in the units they feed it:
/// structural packs whole blocks (splitting oversized ones at sentences) and starts a new
/// chunk at every heading; sentence packs sentences; token-window packs words.
/// Chunks never span a page boundary, so a PDF chunk's page always contains its range.
public enum Chunker {
    public static func chunk(
        _ document: ParsedDocument,
        options: ChunkingOptions = ChunkingOptions(),
        counter: any TokenCounting = ApproximateTokenCounter()
    ) -> [Chunk] {
        let scalars = Array(document.text.unicodeScalars)
        let context = Context(scalars: scalars, blocks: document.blocks, counter: counter, target: options.targetTokens)

        let units: [Unit]
        switch options.kind {
        case .structural: units = context.structuralUnits()
        case .sentence: units = context.sentenceUnits()
        case .tokenWindow: units = context.wordUnits()
        }
        return pack(units, context: context, overlap: options.overlapTokens)
    }

    // MARK: Packing

    struct Unit {
        var range: Range<Int>
        var tokens: Int
        var blockIndex: Int
        var page: Int?
        var startsSection: Bool
    }

    private static func pack(_ units: [Unit], context: Context, overlap: Int) -> [Chunk] {
        var chunks: [Chunk] = []
        var current: [Unit] = []
        var currentTokens = 0
        var fresh = 0  // units in `current` that are not carried over from the previous chunk

        func emit() {
            guard let first = current.first, let last = current.last else { return }
            let range = first.range.lowerBound..<last.range.upperBound
            chunks.append(Chunk(
                ordinal: chunks.count,
                range: range,
                text: String(String.UnicodeScalarView(context.scalars[range])),
                page: first.page,
                headingPath: context.blocks[first.blockIndex].headingPath,
                tokenCount: context.counter.count(context.scalars[range])
            ))
        }

        func flush(carry: Bool) {
            emit()
            var carried: [Unit] = []
            var tokens = 0
            if carry, overlap > 0 {
                // Contiguous trailing units only: stop at the first one that does not fit.
                for unit in current.reversed() {
                    guard tokens + unit.tokens <= overlap else { break }
                    carried.append(unit)
                    tokens += unit.tokens
                }
                carried.reverse()
                if carried.count == current.count { carried.removeFirst() }  // always make progress
            }
            current = carried
            currentTokens = carried.reduce(0) { $0 + $1.tokens }
            fresh = 0
        }

        for unit in units {
            if let last = current.last {
                if last.page != unit.page || unit.startsSection {
                    if fresh > 0 { flush(carry: false) }
                    current = []
                    currentTokens = 0
                } else if currentTokens + unit.tokens > context.target {
                    if fresh > 0 { flush(carry: true) }
                    if currentTokens + unit.tokens > context.target {
                        current = []
                        currentTokens = 0
                    }
                }
            }
            current.append(unit)
            currentTokens += unit.tokens
            fresh += 1
        }
        if fresh > 0 { flush(carry: false) }
        return chunks
    }

    // MARK: Unit generation

    struct Context {
        let scalars: [Unicode.Scalar]
        let blocks: [ParsedBlock]
        let counter: any TokenCounting
        let target: Int

        func tokens(_ range: Range<Int>) -> Int { counter.count(scalars[range]) }

        func structuralUnits() -> [Unit] {
            var units: [Unit] = []
            for (index, block) in blocks.enumerated() {
                var isHeading = false
                if case .heading = block.kind { isHeading = true }
                for (n, range) in fit(block.range, allowSentences: true).enumerated() {
                    units.append(Unit(
                        range: range, tokens: tokens(range), blockIndex: index,
                        page: block.page, startsSection: isHeading && n == 0))
                }
            }
            return units
        }

        func sentenceUnits() -> [Unit] {
            var units: [Unit] = []
            for (index, block) in blocks.enumerated() {
                for sentence in sentences(in: block.range) {
                    for range in fit(sentence, allowSentences: false) {
                        units.append(Unit(
                            range: range, tokens: tokens(range), blockIndex: index,
                            page: block.page, startsSection: false))
                    }
                }
            }
            return units
        }

        func wordUnits() -> [Unit] {
            var units: [Unit] = []
            for (index, block) in blocks.enumerated() {
                for word in words(in: block.range) {
                    for range in fit(word, allowSentences: false) {
                        units.append(Unit(
                            range: range, tokens: tokens(range), blockIndex: index,
                            page: block.page, startsSection: false))
                    }
                }
            }
            return units
        }

        /// Returns `range` whole if it fits the target, otherwise pieces that do:
        /// sentence-packed first, then word-packed, then a hard split for a single huge word.
        func fit(_ range: Range<Int>, allowSentences: Bool) -> [Range<Int>] {
            guard tokens(range) > target else { return [range] }
            if allowSentences {
                let parts = sentences(in: range)
                if parts.count > 1 {
                    return greedy(parts).flatMap { piece in
                        tokens(piece) > target ? fit(piece, allowSentences: false) : [piece]
                    }
                }
            }
            let parts = words(in: range)
            if parts.count > 1 {
                return greedy(parts).flatMap { piece in
                    tokens(piece) > target ? hardSplit(piece) : [piece]
                }
            }
            return hardSplit(range)
        }

        /// Merges consecutive parts while the merged span stays within the target.
        private func greedy(_ parts: [Range<Int>]) -> [Range<Int>] {
            var result: [Range<Int>] = []
            var start: Int?
            var end = 0
            for part in parts {
                if let begin = start {
                    if tokens(begin..<part.upperBound) <= target {
                        end = part.upperBound
                    } else {
                        result.append(begin..<end)
                        start = part.lowerBound
                        end = part.upperBound
                    }
                } else {
                    start = part.lowerBound
                    end = part.upperBound
                }
            }
            if let begin = start { result.append(begin..<end) }
            return result
        }

        private func hardSplit(_ range: Range<Int>) -> [Range<Int>] {
            let step = max(target * 4, 1)
            return stride(from: range.lowerBound, to: range.upperBound, by: step).map {
                $0..<min($0 + step, range.upperBound)
            }
        }

        func words(in range: Range<Int>) -> [Range<Int>] {
            var result: [Range<Int>] = []
            var start: Int?
            for i in range {
                if scalars[i].properties.isWhitespace {
                    if let s = start { result.append(s..<i); start = nil }
                } else if start == nil {
                    start = i
                }
            }
            if let s = start { result.append(s..<range.upperBound) }
            return result
        }

        func sentences(in range: Range<Int>) -> [Range<Int>] {
            let text = String(String.UnicodeScalarView(scalars[range]))
            let tokenizer = NLTokenizer(unit: .sentence)
            tokenizer.string = text
            var result: [Range<Int>] = []
            var cursor = text.unicodeScalars.startIndex
            var offset = 0
            tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { found, _ in
                let scalarView = text.unicodeScalars
                offset += scalarView.distance(from: cursor, to: found.lowerBound)
                let lower = offset
                offset += scalarView.distance(from: found.lowerBound, to: found.upperBound)
                cursor = found.upperBound
                var span = (range.lowerBound + lower)..<(range.lowerBound + offset)
                while span.lowerBound < span.upperBound, scalars[span.lowerBound].properties.isWhitespace {
                    span = (span.lowerBound + 1)..<span.upperBound
                }
                while span.lowerBound < span.upperBound, scalars[span.upperBound - 1].properties.isWhitespace {
                    span = span.lowerBound..<(span.upperBound - 1)
                }
                if !span.isEmpty { result.append(span) }
                return true
            }
            return result.isEmpty ? [range] : result
        }
    }
}
