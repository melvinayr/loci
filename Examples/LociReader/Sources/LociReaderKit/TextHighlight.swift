import Foundation
import SwiftUI

/// A document's text with one cited range marked, for formats that are not PDFs.
public struct HighlightedText: Sendable {
    public let text: AttributedString
    /// The highlighted text, exactly as cited.
    public let highlighted: String
    /// True when text before the window was left out.
    public let startsMidDocument: Bool
    /// True when text after the window was left out.
    public let endsMidDocument: Bool

    /// Marks `scalarRange` (Unicode-scalar offsets, as in `Citation.range`) in `documentText`, showing at most `context` characters
    /// before and after it so the passage is always in view. A range that does not fit the text highlights nothing and shows the
    /// beginning of the document.
    public static func make(documentText: String, scalarRange: Range<Int>, context: Int = 1_500) -> HighlightedText {
        let scalars = documentText.unicodeScalars
        guard scalarRange.lowerBound >= 0, scalarRange.upperBound <= scalars.count, !scalarRange.isEmpty else {
            let end = scalars.index(scalars.startIndex, offsetBy: min(context * 2, scalars.count))
            return HighlightedText(
                text: AttributedString(String(scalars[scalars.startIndex..<end])), highlighted: "", startsMidDocument: false,
                endsMidDocument: end < scalars.endIndex)
        }
        let windowStart = max(0, scalarRange.lowerBound - context)
        let windowEnd = min(scalars.count, scalarRange.upperBound + context)
        let start = scalars.index(scalars.startIndex, offsetBy: scalarRange.lowerBound)
        let end = scalars.index(start, offsetBy: scalarRange.count)
        let from = scalars.index(scalars.startIndex, offsetBy: windowStart)
        let to = scalars.index(from, offsetBy: windowEnd - windowStart)

        let window = String(scalars[from..<to])
        var attributed = AttributedString(window)
        let windowScalars = window.unicodeScalars
        let inner = windowScalars.index(windowScalars.startIndex, offsetBy: scalarRange.lowerBound - windowStart)
        let innerEnd = windowScalars.index(inner, offsetBy: scalarRange.count)
        if let lower = AttributedString.Index(inner, within: attributed), let upper = AttributedString.Index(innerEnd, within: attributed) {
            attributed[lower..<upper].backgroundColor = .yellow.opacity(0.45)
        }
        return HighlightedText(
            text: attributed, highlighted: String(scalars[start..<end]), startsMidDocument: windowStart > 0, endsMidDocument: windowEnd < scalars.count)
    }
}
