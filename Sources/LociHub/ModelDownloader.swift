import CryptoKit
import Foundation
import Loci

/// Downloads GGUF models from Hugging Face into the folder `KnowledgeBase` looks in for `.hub` sources.
///
/// Downloads resume from a partial file, are checked against the SHA-256 in the `ModelSource` (a corrupt file is deleted, never
/// kept), and are excluded from backups. This is the only part of Loci that uses the network.
public actor ModelDownloader {
    public nonisolated let directory: URL
    private let baseURL: URL
    private let configuration: URLSessionConfiguration
    private let availableSpace: @Sendable (URL) -> Int64?
    private var inFlight: [String: (task: Task<URL, Error>, waiters: Int)] = [:]

    /// The Hugging Face host.
    public static let huggingFace = URL(string: "https://huggingface.co") ?? URL(fileURLWithPath: "/")

    /// Application Support/Loci/Models, where `ModelSet` resolves `.hub` sources.
    public static var defaultDirectory: URL {
        let support = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        return support.appendingPathComponent("Loci", isDirectory: true).appendingPathComponent("Models", isDirectory: true)
    }

    /// - Parameters:
    ///   - directory: where models are stored; leave the default so `KnowledgeBase` finds them.
    ///   - configuration: the URL session settings (tests inject a stub).
    ///   - baseURL: the Hugging Face host.
    ///   - availableSpace: free bytes at a location; overridable for tests.
    public init(
        directory: URL = ModelDownloader.defaultDirectory,
        configuration: URLSessionConfiguration = .default,
        baseURL: URL = ModelDownloader.huggingFace,
        availableSpace: (@Sendable (URL) -> Int64?)? = nil
    ) {
        self.directory = directory
        self.configuration = configuration
        self.baseURL = baseURL
        self.availableSpace = availableSpace ?? { url in
            (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?.volumeAvailableCapacityForImportantUsage
        }
    }

    // MARK: Locating

    /// The file for `source` if it is on disk and passes verification; nil otherwise. Never touches the network.
    public func localURL(for source: ModelSource) async -> URL? {
        guard case .hub(_, let file, let sha256) = source, Self.isPlainFileName(file) else { return nil }
        let url = directory.appendingPathComponent(file)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return Self.verified(url, sha256: sha256) ? url : nil
    }

    /// True when `localURL(for:)` would find the file.
    public func isDownloaded(_ source: ModelSource) async -> Bool {
        await localURL(for: source) != nil
    }

    /// Deletes a downloaded model and any partial download of it.
    public func remove(_ source: ModelSource) throws {
        guard case .hub(_, let file, _) = source, Self.isPlainFileName(file) else { return }
        for url in [directory.appendingPathComponent(file), Self.partURL(directory, file), Self.markerURL(directory, file)] {
            if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
        }
    }

    // MARK: Downloading

    /// Makes sure the model is on disk and verified, downloading it if needed, and returns its location.
    ///
    /// Concurrent calls for the same file share one download.
    @discardableResult
    public func ensure(_ source: ModelSource, progress: (@Sendable (DownloadProgress) -> Void)? = nil) async throws -> URL {
        guard case .hub(let repo, let file, let sha256) = source else {
            throw ModelHubError.invalidSource("only Hugging Face (.hub) models can be downloaded")
        }
        try Self.validate(repo: repo, file: file)
        if let existing = await localURL(for: source) { return existing }

        // Callers share one download. It is cancelled only when every caller waiting for it has been cancelled.
        let task: Task<URL, Error>
        if let running = inFlight[file] {
            task = running.task
            inFlight[file] = (task, running.waiters + 1)
        } else {
            task = Task { try await self.fetch(repo: repo, file: file, sha256: sha256, progress: progress) }
            inFlight[file] = (task, 1)
        }
        defer { finishWaiting(file, task) }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            Task { await self.cancelWaiter(file, task) }
        }
    }

    private func cancelWaiter(_ file: String, _ task: Task<URL, Error>) {
        guard let running = inFlight[file], running.task == task else { return }
        if running.waiters <= 1 { task.cancel() }
    }

    private func finishWaiting(_ file: String, _ task: Task<URL, Error>) {
        guard let running = inFlight[file], running.task == task else { return }
        if running.waiters <= 1 { inFlight[file] = nil } else { inFlight[file] = (task, running.waiters - 1) }
    }

    /// Like `ensure`, as a stream: progress events, then `.finished` with the file's location.
    public nonisolated func download(_ source: ModelSource) -> AsyncThrowingStream<DownloadEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let url = try await self.ensure(source) { continuation.yield(.progress($0)) }
                    continuation.yield(.finished(url))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: Private

    private func fetch(repo: String, file: String, sha256: String?, progress: (@Sendable (DownloadProgress) -> Void)?) async throws -> URL {
        let manager = FileManager.default
        do {
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw ModelHubError.io(error.localizedDescription)
        }
        let destination = directory.appendingPathComponent(file)
        let part = Self.partURL(directory, file)
        // A stale unverified file or marker is replaced.
        try? manager.removeItem(at: destination)
        try? manager.removeItem(at: Self.markerURL(directory, file))

        let url = try Self.resolveURL(base: baseURL, repo: repo, file: file)
        var attempt = 0
        while true {
            attempt += 1
            let offset = Self.state(of: part)?.size ?? 0
            var request = URLRequest(url: url)
            request.timeoutInterval = 60
            if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }

            let directory = self.directory
            let space = availableSpace
            let session = DownloadSession(
                request: request, part: part, offset: offset,
                spaceCheck: { needed in
                    // Leave 64 MB spare so the device is not filled to the brim.
                    guard let available = space(directory), available < needed + (64 << 20) else { return nil }
                    return .insufficientDiskSpace(required: needed + (64 << 20), available: available)
                },
                onProgress: { progress?($0) })
            let outcome = try await session.run(configuration: configuration)
            switch outcome {
            case .completed(let received, let total):
                if let total, received != total { throw ModelHubError.network("the connection ended after \(received) of \(total) bytes") }
            case .rangeNotSatisfiable:
                // The part file is not usable (already whole, or from a different file): start again once.
                try? manager.removeItem(at: part)
                if attempt >= 2 { throw ModelHubError.httpStatus(416) }
                continue
            }
            break
        }

        if let sha256 {
            let actual = try Self.sha256Hex(of: part)
            guard actual.lowercased() == sha256.lowercased() else {
                try? manager.removeItem(at: part)
                throw ModelHubError.checksumMismatch(expected: sha256.lowercased(), actual: actual)
            }
        }
        do {
            try manager.moveItem(at: part, to: destination)
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var excluded = destination
            try? excluded.setResourceValues(values)
            Self.writeMarker(for: destination, sha256: sha256, directory: directory, file: file)
        } catch {
            throw ModelHubError.io(error.localizedDescription)
        }
        return destination
    }

    // MARK: Helpers

    static func isPlainFileName(_ file: String) -> Bool {
        !file.isEmpty && !file.contains("/") && !file.contains("\\") && file != "." && file != ".." && !file.hasPrefix(".") && !file.contains("\0")
    }

    static func validate(repo: String, file: String) throws {
        let parts = repo.split(separator: "/", omittingEmptySubsequences: false)
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && $0.unicodeScalars.allSatisfy(allowed.contains) && !$0.hasPrefix(".") }) else {
            throw ModelHubError.invalidSource("“\(repo)” is not a Hugging Face repository (owner/name)")
        }
        guard isPlainFileName(file) else { throw ModelHubError.invalidSource("“\(file)” is not a plain file name") }
    }

    static func resolveURL(base: URL, repo: String, file: String) throws -> URL {
        var components = URLComponents(url: base, resolvingAgainstBaseURL: false)
        let encodedFile = file.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file
        components?.percentEncodedPath = "/\(repo)/resolve/main/\(encodedFile)"
        guard let url = components?.url else { throw ModelHubError.invalidSource("the download address could not be built") }
        return url
    }

    static func partURL(_ directory: URL, _ file: String) -> URL { directory.appendingPathComponent(file + ".part") }
    static func markerURL(_ directory: URL, _ file: String) -> URL { directory.appendingPathComponent(file + ".verified") }

    /// A completed file is trusted without re-hashing gigabytes while its size and date still match what was verified.
    private struct Marker: Codable, Equatable {
        var sha256: String
        var size: Int64
        var modified: TimeInterval
    }

    static func writeMarker(for file: URL, sha256: String?, directory: URL, file name: String) {
        guard let sha256, let state = state(of: file) else { return }
        let marker = Marker(sha256: sha256.lowercased(), size: state.size, modified: state.modified)
        if let data = try? JSONEncoder().encode(marker) { try? data.write(to: markerURL(directory, name)) }
    }

    /// Size and modification date straight from the file system. (`URL.resourceValues` caches, and would report a
    /// file that has since been deleted or rewritten as it used to be.)
    private static func state(of url: URL) -> (size: Int64, modified: TimeInterval)? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = (attributes[.size] as? NSNumber)?.int64Value, let modified = attributes[.modificationDate] as? Date else { return nil }
        return (size, modified.timeIntervalSince1970)
    }

    /// Whether `url` is the file `sha256` describes. Without a checksum there is nothing to verify and an existing file is accepted.
    static func verified(_ url: URL, sha256: String?) -> Bool {
        guard let sha256 else { return true }
        let directory = url.deletingLastPathComponent()
        if let data = try? Data(contentsOf: markerURL(directory, url.lastPathComponent)),
           let marker = try? JSONDecoder().decode(Marker.self, from: data), let current = state(of: url),
           marker.sha256 == sha256.lowercased(), marker.size == current.size, marker.modified == current.modified {
            return true
        }
        guard let actual = try? sha256Hex(of: url), actual.lowercased() == sha256.lowercased() else { return false }
        writeMarker(for: url, sha256: sha256, directory: directory, file: url.lastPathComponent)
        return true
    }

    /// Streams the file through SHA-256 in 4 MB pieces.
    static func sha256Hex(of url: URL) throws -> String {
        let handle: FileHandle
        do { handle = try FileHandle(forReadingFrom: url) } catch { throw ModelHubError.io(error.localizedDescription) }
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk: Data
            do { chunk = try handle.read(upToCount: 4 << 20) ?? Data() } catch { throw ModelHubError.io(error.localizedDescription) }
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
