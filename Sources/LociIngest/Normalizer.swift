import Foundation

/// Text normalisation applied to every block before it is hashed, stored or offset.
///
/// Offsets are only ever computed on normalised text, so they stay stable.
public enum Normalizer {
    /// Strips invisible and control scalars, applies Unicode NFC, collapses every
    /// whitespace run to a single space and trims. Idempotent.
    public static func normalize(_ raw: String) -> String {
        var kept = String.UnicodeScalarView()
        for scalar in raw.unicodeScalars where !isStripped(scalar) {
            kept.append(scalar)
        }
        let composed = String(kept).precomposedStringWithCanonicalMapping

        var output = String.UnicodeScalarView()
        var pendingSpace = false
        for scalar in composed.unicodeScalars {
            if isWhitespace(scalar) {
                pendingSpace = !output.isEmpty
            } else {
                if pendingSpace { output.append(" ") }
                pendingSpace = false
                output.append(scalar)
            }
        }
        return String(output)
    }

    static func isWhitespace(_ scalar: Unicode.Scalar) -> Bool {
        scalar.properties.isWhitespace
    }

    /// Soft hyphen, zero-width characters, BOM and non-whitespace control characters.
    private static func isStripped(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x00AD, 0x200B...0x200D, 0x2060, 0xFEFF: return true
        default:
            return scalar.properties.generalCategory == .control && !scalar.properties.isWhitespace
        }
    }
}
