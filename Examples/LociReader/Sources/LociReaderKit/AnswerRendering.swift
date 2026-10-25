import Foundation
import SwiftUI

/// Turns an answer sentence into text whose `[n]` markers are tappable.
public enum AnswerRendering {
    public static let scheme = "loci-citation"

    /// `text` with every `[n]` marker made a link to citation `n`.
    public static func attributed(_ text: String) -> AttributedString {
        var result = AttributedString()
        var cursor = text.startIndex
        guard let regex = try? NSRegularExpression(pattern: #"\[(\d{1,3})\]"#) else { return AttributedString(text) }
        for match in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let whole = Range(match.range, in: text), let digits = Range(match.range(at: 1), in: text) else { continue }
            result += AttributedString(String(text[cursor..<whole.lowerBound]))
            var marker = AttributedString(String(text[whole]))
            marker.link = URL(string: "\(scheme)://\(text[digits])")
            marker.foregroundColor = .accentColor
            result += marker
            cursor = whole.upperBound
        }
        result += AttributedString(String(text[cursor...]))
        return result
    }

    /// The citation number a tapped link refers to.
    public static func citationIndex(from url: URL) -> Int? {
        url.scheme == scheme ? url.host.flatMap(Int.init) : nil
    }
}
