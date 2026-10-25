import Foundation

/// Turns a stream of token byte pieces into text without ever emitting a partial code point.
///
/// A token can end in the middle of a multi-byte character; the incomplete tail is held back until
/// the next piece completes it.
struct UTF8StreamDecoder {
    private var pending: [UInt8] = []

    /// Appends `bytes` and returns the text that is now complete.
    mutating func decode(_ bytes: [UInt8]) -> String {
        pending += bytes
        let complete = Self.completePrefixLength(of: pending)
        guard complete > 0 else { return "" }
        let text = String(decoding: pending[..<complete], as: UTF8.self)
        pending.removeFirst(complete)
        return text
    }

    /// Whatever is left at the end of the stream; an unfinished character becomes U+FFFD.
    mutating func finish() -> String {
        defer { pending.removeAll() }
        return pending.isEmpty ? "" : String(decoding: pending, as: UTF8.self)
    }

    /// Length of the longest prefix that does not end inside a multi-byte sequence.
    static func completePrefixLength(of bytes: [UInt8]) -> Int {
        // An incomplete sequence can only start within the last 3 bytes.
        let count = bytes.count
        var offset = 1
        while offset <= min(3, count) {
            let byte = bytes[count - offset]
            if byte & 0b1100_0000 == 0b1000_0000 {  // continuation byte: keep looking for its lead
                offset += 1
                continue
            }
            let expected = byte >= 0xF0 ? 4 : byte >= 0xE0 ? 3 : byte >= 0xC0 ? 2 : 1
            return expected > offset ? count - offset : count
        }
        return count
    }
}
