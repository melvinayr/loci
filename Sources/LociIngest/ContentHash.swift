import CryptoKit
import Foundation

/// SHA-256 of normalised document text, as lowercase hex.
public enum ContentHash {
    public static func sha256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}
