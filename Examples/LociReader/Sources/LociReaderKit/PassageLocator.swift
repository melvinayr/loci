import Foundation

/// Finds a cited passage inside the text of one PDF page.
///
/// Loci's excerpts are normalised (NFC, whitespace collapsed), while the text PDFKit extracts keeps line breaks, ligatures such as
/// "ﬁ", and sometimes hyphenation. Matching therefore ignores everything but letters and digits, folds case and compatibility
/// forms, and maps the match back to UTF-16 offsets in the page's own text, which is what PDFKit selections use.
public enum PassageLocator {
    /// The UTF-16 range in `pageText` of the text matching `excerpt`, or nil if it is not found.
    ///
    /// If the whole excerpt does not match (for instance because of an OCR slip), the start and end of the excerpt are
    /// located separately and the range between them is returned, provided they are in the right order and close together.
    public static func locate(excerpt: String, inPageText pageText: String) -> NSRange? {
        let page = Folded(pageText)
        let wanted = Folded(excerpt)
        guard !wanted.characters.isEmpty, !page.characters.isEmpty else { return nil }

        if let range = page.find(wanted.characters, from: 0) {
            return extendedOverClosingPunctuation(page.nsRange(of: range), pageText: pageText, excerpt: excerpt)
        }

        // Fall back to the beginning and the end of the excerpt.
        let edge = min(40, wanted.characters.count)
        guard wanted.characters.count > edge * 2 else { return nil }
        let head = Array(wanted.characters.prefix(edge)), tail = Array(wanted.characters.suffix(edge))
        guard let start = page.find(head, from: 0), let end = page.find(tail, from: start.lowerBound) else { return nil }
        let span = end.upperBound - start.lowerBound
        guard span <= wanted.characters.count * 2 else { return nil }
        return page.nsRange(of: start.lowerBound..<end.upperBound)
    }

    /// Folding drops punctuation, so a match ends at its last letter; if the excerpt ends in punctuation, take the same
    /// punctuation from the page so the highlight covers the whole sentence.
    private static func extendedOverClosingPunctuation(_ range: NSRange, pageText: String, excerpt: String) -> NSRange {
        let trimmed = excerpt.trimmingCharacters(in: .whitespacesAndNewlines)
        var wanted = 0
        for character in trimmed.reversed() {
            guard character.isPunctuation || character.isSymbol else { break }
            wanted += 1
        }
        guard wanted > 0 else { return range }
        let page = pageText as NSString
        var end = range.location + range.length
        var taken = 0
        while taken < wanted, end < page.length {
            let character = page.substring(with: page.rangeOfComposedCharacterSequence(at: end))
            guard character.allSatisfy({ $0.isPunctuation || $0.isSymbol }) else { break }
            end += character.utf16.count
            taken += 1
        }
        return NSRange(location: range.location, length: end - range.location)
    }

    /// Lowercase letters and digits of a text, each remembering where it came from.
    private struct Folded {
        var characters: [Character] = []
        /// UTF-16 offset and length in the original text for each entry of `characters`.
        var origins: [(offset: Int, length: Int)] = []

        init(_ text: String) {
            var offset = 0
            for character in text {
                let length = character.utf16.count
                // NFKC turns a ligature into its letters; each of them points at the one original character.
                for folded in String(character).decomposedStringWithCompatibilityMapping.lowercased() where folded.isLetter || folded.isNumber {
                    characters.append(folded)
                    origins.append((offset, length))
                }
                offset += length
            }
        }

        func find(_ needle: [Character], from start: Int) -> Range<Int>? {
            guard !needle.isEmpty, characters.count >= needle.count, start <= characters.count - needle.count else { return nil }
            let first = needle[0]
            var index = start
            while index <= characters.count - needle.count {
                if characters[index] == first, characters[index..<index + needle.count].elementsEqual(needle) {
                    return index..<index + needle.count
                }
                index += 1
            }
            return nil
        }

        func nsRange(of range: Range<Int>) -> NSRange {
            let first = origins[range.lowerBound]
            let last = origins[range.upperBound - 1]
            return NSRange(location: first.offset, length: last.offset + last.length - first.offset)
        }
    }
}
