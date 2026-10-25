import Foundation

/// One HTTP download written straight to a file.
///
/// Uses a data-task delegate because iterating `URLSession.AsyncBytes` yields one byte at a time, which is far too slow
/// for multi-gigabyte models. All state is touched only from the session's serial delegate queue.
final class DownloadSession: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    // Safety of @unchecked Sendable: `handle`, `received`, `total` and `failure` are only read or written inside the
    // delegate callbacks, which URLSession runs one at a time on `queue`; `task` is guarded by `lock`.
    enum Outcome: Sendable {
        case completed(received: Int64, total: Int64?)
        /// The server cannot serve the range we asked for (the part file is already complete or stale).
        case rangeNotSatisfiable
    }

    private let request: URLRequest
    private let part: URL
    private let offset: Int64
    private let onProgress: @Sendable (DownloadProgress) -> Void
    private let spaceCheck: @Sendable (_ bytesToWrite: Int64) -> ModelHubError?
    private let queue = OperationQueue()
    private var session: URLSession?
    private var handle: FileHandle?
    private var received: Int64
    private var total: Int64?
    private var failure: ModelHubError?
    private var rangeRefused = false
    private var continuation: CheckedContinuation<Outcome, Error>?
    private let lock = NSLock()
    private var task: URLSessionDataTask?
    private var lastReported: Int64 = 0

    /// - Parameter spaceCheck: called once the response says how many bytes are coming; returns an error to stop the download.
    init(
        request: URLRequest, part: URL, offset: Int64, spaceCheck: @escaping @Sendable (_ bytesToWrite: Int64) -> ModelHubError?,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void
    ) {
        self.request = request
        self.spaceCheck = spaceCheck
        self.part = part
        self.offset = offset
        self.received = offset
        self.onProgress = onProgress
        queue.maxConcurrentOperationCount = 1
    }

    func run(configuration: URLSessionConfiguration) async throws -> Outcome {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                self.continuation = continuation
                let session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
                self.session = session
                let task = session.dataTask(with: request)
                lock.withLock { self.task = task }
                task.resume()
            }
        } onCancel: {
            lock.withLock { task }?.cancel()
        }
    }

    // MARK: URLSessionDataDelegate

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            failure = .network("the server sent an unexpected response")
            completionHandler(.cancel)
            return
        }
        switch http.statusCode {
        case 206:
            // Resuming: the server must continue exactly where the part file ends.
            guard let range = http.value(forHTTPHeaderField: "Content-Range"), Self.start(of: range) == offset else {
                failure = .network("the server resumed from the wrong position")
                completionHandler(.cancel)
                return
            }
            total = Self.total(of: range)
            if let total, let error = spaceCheck(total - offset) {
                failure = error
                completionHandler(.cancel)
                return
            }
            openPart(truncate: false)
        case 200:
            // Either a fresh download or a server that ignored our Range: start the file over.
            received = 0
            lastReported = 0
            total = http.expectedContentLength > 0 ? http.expectedContentLength : nil
            if let total, let error = spaceCheck(total) {
                failure = error
                completionHandler(.cancel)
                return
            }
            openPart(truncate: true)
        case 416:
            rangeRefused = true
            completionHandler(.cancel)
            return
        default:
            failure = .httpStatus(http.statusCode)
            completionHandler(.cancel)
            return
        }
        completionHandler(failure == nil ? .allow : .cancel)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard failure == nil, let handle else { return }
        do {
            try handle.write(contentsOf: data)
            received += Int64(data.count)
            // Report about every 256 KB so a fast download does not flood the consumer.
            if received - lastReported >= 256 * 1024 || received == total {
                lastReported = received
                onProgress(DownloadProgress(receivedBytes: received, totalBytes: total))
            }
        } catch {
            failure = .io(error.localizedDescription)
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        try? handle?.close()
        handle = nil
        session.finishTasksAndInvalidate()
        guard let continuation else { return }
        self.continuation = nil

        if let failure {
            continuation.resume(throwing: failure)
        } else if rangeRefused {
            continuation.resume(returning: .rangeNotSatisfiable)
        } else if let error = error as? URLError, error.code == .cancelled {
            continuation.resume(throwing: CancellationError())
        } else if let error {
            continuation.resume(throwing: ModelHubError.network(error.localizedDescription))
        } else {
            onProgress(DownloadProgress(receivedBytes: received, totalBytes: total ?? received))
            continuation.resume(returning: .completed(received: received, total: total))
        }
    }

    // MARK: Helpers

    private func openPart(truncate: Bool) {
        do {
            let manager = FileManager.default
            if truncate || !manager.fileExists(atPath: part.path) {
                guard manager.createFile(atPath: part.path, contents: nil) else { throw CocoaError(.fileWriteUnknown) }
            }
            let handle = try FileHandle(forWritingTo: part)
            try handle.seekToEnd()
            self.handle = handle
        } catch {
            failure = .io(error.localizedDescription)
        }
    }

    /// `bytes 100-199/1000` → 100.
    static func start(of contentRange: String) -> Int64? {
        let parts = contentRange.split(whereSeparator: { $0 == " " || $0 == "-" || $0 == "/" })
        return parts.count >= 2 ? Int64(parts[1]) : nil
    }

    /// `bytes 100-199/1000` → 1000 (nil for `*`).
    static func total(of contentRange: String) -> Int64? {
        contentRange.split(separator: "/").last.flatMap { Int64($0) }
    }
}
