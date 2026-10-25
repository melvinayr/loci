import Foundation

enum FileProtection {
    /// Applies `type` to the file at `url` if it exists. Data Protection is an iOS
    /// feature; on macOS this is a no-op.
    static func apply(_ type: FileProtectionType, to url: URL) throws {
        #if os(iOS)
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            try FileManager.default.setAttributes([.protectionKey: type], ofItemAtPath: url.path)
        } catch {
            throw StoreError.corrupted("file protection could not be applied")
        }
        #endif
    }
}
