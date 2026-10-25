import CryptoKit
import Foundation
import os
import Testing
@testable import Loci
@testable import LociHub

/// A fake Hugging Face. Each test uses its own host so tests can run in parallel.
final class StubServer: Sendable {
    struct Reply {
        var status = 200
        var headers: [String: String] = [:]
        var body = Data()
    }

    let host = "stub-\(UUID().uuidString.prefix(8).lowercased()).test"
    private let handler: @Sendable (URLRequest) -> Reply
    private let state = OSAllocatedUnfairLock(initialState: (requests: [URLRequest](), chunkDelay: Duration?.none))

    static let registry = OSAllocatedUnfairLock(initialState: [String: StubServer]())

    init(handler: @escaping @Sendable (URLRequest) -> Reply) {
        self.handler = handler
        Self.registry.withLock { $0[host] = self }
    }

    deinit { Self.registry.withLock { _ = $0.removeValue(forKey: host) } }

    var requests: [URLRequest] { state.withLock { $0.requests } }
    var chunkDelay: Duration? {
        get { state.withLock { $0.chunkDelay } }
        set { state.withLock { $0.chunkDelay = newValue } }
    }

    var baseURL: URL { URL(string: "https://\(host)") ?? URL(fileURLWithPath: "/") }

    var configuration: URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubProtocol.self]
        return configuration
    }

    func reply(to request: URLRequest) -> Reply {
        state.withLock { $0.requests.append(request) }
        return handler(request)
    }

    /// Serves `data` the way Hugging Face does, honouring `Range: bytes=N-` unless `ignoreRange`.
    static func serving(_ data: Data, ignoreRange: Bool = false) -> StubServer {
        StubServer { request in
            if !ignoreRange, let range = request.value(forHTTPHeaderField: "Range"), range.hasPrefix("bytes=") {
                let start = Int(range.dropFirst(6).split(separator: "-").first ?? "") ?? 0
                if start >= data.count { return Reply(status: 416, headers: ["Content-Range": "bytes */\(data.count)"]) }
                return Reply(status: 206, headers: ["Content-Range": "bytes \(start)-\(data.count - 1)/\(data.count)"], body: Data(data.suffix(from: start)))
            }
            return Reply(status: 200, headers: ["Content-Length": "\(data.count)"], body: data)
        }
    }
}

final class StubProtocol: URLProtocol, @unchecked Sendable {
    private let stopped = OSAllocatedUnfairLock(initialState: false)

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let host = request.url?.host, let server = StubServer.registry.withLock({ $0[host] }), let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
            return
        }
        let reply = server.reply(to: request)
        let delay = server.chunkDelay
        DispatchQueue.global().async { [self] in
            guard let response = HTTPURLResponse(url: url, statusCode: reply.status, httpVersion: "HTTP/1.1", headerFields: reply.headers) else { return }
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            var offset = 0
            while offset < reply.body.count {
                if stopped.withLock({ $0 }) { return }
                if let delay { Thread.sleep(forTimeInterval: Double(delay.components.seconds) + Double(delay.components.attoseconds) / 1e18) }
                let end = min(offset + 64 * 1024, reply.body.count)
                client?.urlProtocol(self, didLoad: reply.body.subdata(in: offset..<end))
                offset = end
            }
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    override func stopLoading() { stopped.withLock { $0 = true } }
}

private func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

private func makeData(_ count: Int = 600_000) -> Data {
    var generator = SeededGenerator(seed: 7)
    return Data((0..<count).map { _ in UInt8.random(in: 0...255, using: &generator) })
}

private func withDownloader(
    server: StubServer, freeSpace: Int64? = nil, _ body: (ModelDownloader, URL) async throws -> Void
) async throws {
    let directory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let models = directory.appendingPathComponent("Models")
    let downloader = ModelDownloader(
        directory: models, configuration: server.configuration, baseURL: server.baseURL,
        availableSpace: freeSpace.map { value in { @Sendable _ in value } })
    try await body(downloader, models)
}

private func source(_ data: Data?, file: String = "model.gguf") -> ModelSource {
    .hub(repo: "owner/model-GGUF", file: file, sha256: data.map(sha256))
}

@Suite("Model downloads")
struct ModelDownloaderTests {
    @Test func downloadsVerifiesAndReportsMonotonicProgress() async throws {
        let data = makeData()
        let server = StubServer.serving(data)
        try await withDownloader(server: server) { downloader, models in
            let reports = OSAllocatedUnfairLock(initialState: [DownloadProgress]())
            let url = try await downloader.ensure(source(data)) { progress in reports.withLock { $0.append(progress) } }

            #expect(url == models.appendingPathComponent("model.gguf"))
            #expect(try Data(contentsOf: url) == data)
            #expect(!FileManager.default.fileExists(atPath: url.path + ".part"), "the partial file is gone")
            let seen = reports.withLock { $0 }
            #expect(seen.count > 1 && seen.map(\.receivedBytes) == seen.map(\.receivedBytes).sorted())
            #expect(seen.last?.receivedBytes == Int64(data.count) && seen.last?.totalBytes == Int64(data.count) && seen.last?.fraction == 1)
            #expect(server.requests.first?.url?.path == "/owner/model-GGUF/resolve/main/model.gguf")
            #expect(server.requests.first?.value(forHTTPHeaderField: "Range") == nil)
            #expect(try url.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup == true)
        }
    }

    @Test func aFileThatIsAlreadyVerifiedIsNotFetchedAgain() async throws {
        let data = makeData()
        let server = StubServer.serving(data)
        try await withDownloader(server: server) { downloader, _ in
            _ = try await downloader.ensure(source(data))
            #expect(server.requests.count == 1)
            _ = try await downloader.ensure(source(data))
            let located = await downloader.localURL(for: source(data))
            #expect(server.requests.count == 1 && located != nil)
            #expect(await downloader.isDownloaded(source(data)))
        }
    }

    @Test func aFileChangedOnDiskFailsVerificationAndIsDownloadedAgain() async throws {
        let data = makeData()
        let server = StubServer.serving(data)
        try await withDownloader(server: server) { downloader, models in
            let url = try await downloader.ensure(source(data))
            var corrupted = data
            corrupted[100] ^= 0xFF
            try corrupted.write(to: url)  // same size, different bytes (and a new modification date)
            #expect(await downloader.localURL(for: source(data)) == nil)
            _ = try await downloader.ensure(source(data))
            #expect(try Data(contentsOf: models.appendingPathComponent("model.gguf")) == data)
            #expect(server.requests.count == 2)
        }
    }

    @Test func aChecksumMismatchDeletesTheDownloadAndThrows() async throws {
        let data = makeData()
        let server = StubServer.serving(data)
        try await withDownloader(server: server) { downloader, models in
            let wrong = ModelSource.hub(repo: "owner/model-GGUF", file: "model.gguf", sha256: String(repeating: "0", count: 64))
            do {
                _ = try await downloader.ensure(wrong)
                Issue.record("expected a checksum error")
            } catch ModelHubError.checksumMismatch(let expected, let actual) {
                #expect(expected == String(repeating: "0", count: 64) && actual == sha256(data))
            }
            let leftovers = (try? FileManager.default.contentsOfDirectory(atPath: models.path)) ?? []
            #expect(leftovers.isEmpty, "nothing corrupt is kept: \(leftovers)")
        }
    }

    @Test func withoutAChecksumTheFileIsAcceptedAsDownloaded() async throws {
        let data = makeData(50_000)
        try await withDownloader(server: .serving(data)) { downloader, _ in
            let url = try await downloader.ensure(source(nil))
            #expect(try Data(contentsOf: url) == data)
        }
    }

    @Test func aPartialFileIsResumedWithARangeRequest() async throws {
        let data = makeData()
        let server = StubServer.serving(data)
        try await withDownloader(server: server) { downloader, models in
            try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
            try data.prefix(250_000).write(to: models.appendingPathComponent("model.gguf.part"))

            let first = OSAllocatedUnfairLock(initialState: Int64?.none)
            let url = try await downloader.ensure(source(data)) { progress in first.withLock { if $0 == nil { $0 = progress.receivedBytes } } }
            #expect(try Data(contentsOf: url) == data)
            #expect(server.requests.first?.value(forHTTPHeaderField: "Range") == "bytes=250000-")
            #expect((first.withLock { $0 } ?? 0) >= 250_000, "progress counts the bytes already on disk")
        }
    }

    @Test func aServerThatIgnoresRangeRestartsTheFileInsteadOfCorruptingIt() async throws {
        let data = makeData()
        let server = StubServer.serving(data, ignoreRange: true)
        try await withDownloader(server: server) { downloader, models in
            try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
            try data.prefix(100_000).write(to: models.appendingPathComponent("model.gguf.part"))
            let url = try await downloader.ensure(source(data))
            #expect(try Data(contentsOf: url) == data, "the part was discarded, not appended to")
        }
    }

    @Test func anOversizedOrStalePartFileTriggersAFreshDownload() async throws {
        let data = makeData(100_000)
        let server = StubServer.serving(data)
        try await withDownloader(server: server) { downloader, models in
            try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
            try (data + Data(repeating: 1, count: 5_000)).write(to: models.appendingPathComponent("model.gguf.part"))  // longer than the real file
            let url = try await downloader.ensure(source(data))
            #expect(try Data(contentsOf: url) == data)
            #expect(server.requests.count == 2 && server.requests[1].value(forHTTPHeaderField: "Range") == nil)
        }
    }

    @Test(arguments: [404, 403, 500])
    func httpErrorsAreReportedAndLeaveNothingBehind(status: Int) async throws {
        let server = StubServer { _ in .init(status: status) }
        try await withDownloader(server: server) { downloader, models in
            do {
                _ = try await downloader.ensure(source(makeData(10)))
                Issue.record("expected an error")
            } catch ModelHubError.httpStatus(let code) {
                #expect(code == status)
                #expect(ModelHubError.httpStatus(code).errorDescription?.isEmpty == false)
            }
            #expect(!FileManager.default.fileExists(atPath: models.appendingPathComponent("model.gguf").path))
        }
    }

    @Test func aTruncatedTransferIsAnErrorAndTheCheckIsNotSkipped() async throws {
        let data = makeData(100_000)
        let server = StubServer { _ in .init(status: 200, headers: ["Content-Length": "100000"], body: Data(data.prefix(40_000))) }
        try await withDownloader(server: server) { downloader, models in
            // CFNetwork reports a short body as a network error; either way the file must not be accepted.
            await #expect(throws: ModelHubError.self) { _ = try await downloader.ensure(source(data)) }
            #expect(!FileManager.default.fileExists(atPath: models.appendingPathComponent("model.gguf").path))
        }
    }

    @Test func notEnoughDiskSpaceIsRefusedBeforeWriting() async throws {
        let data = makeData()
        try await withDownloader(server: .serving(data), freeSpace: 1_000) { downloader, models in
            do {
                _ = try await downloader.ensure(source(data))
                Issue.record("expected insufficientDiskSpace")
            } catch ModelHubError.insufficientDiskSpace(let required, let available) {
                #expect(required > Int64(data.count) && available == 1_000)
            }
            let part = models.appendingPathComponent("model.gguf.part")
            let written = ((try? FileManager.default.attributesOfItem(atPath: part.path)[.size]) as? NSNumber)?.intValue ?? 0
            #expect(written == 0, "nothing was written")
        }
    }

    @Test func cancellingKeepsThePartialFileSoTheNextAttemptResumes() async throws {
        let data = makeData(2_000_000)
        let server = StubServer.serving(data)
        server.chunkDelay = .milliseconds(15)
        try await withDownloader(server: server) { downloader, models in
            let task = Task { try await downloader.ensure(source(data)) }
            try await Task.sleep(for: .milliseconds(300))
            task.cancel()
            await #expect(throws: (any Error).self) { _ = try await task.value }

            let part = models.appendingPathComponent("model.gguf.part")
            let kept = ((try? FileManager.default.attributesOfItem(atPath: part.path)[.size]) as? NSNumber)?.intValue ?? 0
            #expect(kept > 0 && kept < data.count, "kept \(kept) bytes")

            server.chunkDelay = nil
            let url = try await downloader.ensure(source(data))
            #expect(try Data(contentsOf: url) == data)
            let range = try #require(server.requests.last?.value(forHTTPHeaderField: "Range"))
            let resumedFrom = try #require(Int(range.dropFirst(6).dropLast()))
            #expect(resumedFrom > 0 && resumedFrom <= kept + 64 * 1024, "resumed from \(resumedFrom) with \(kept) kept")
        }
    }

    @Test func concurrentRequestsForTheSameFileShareOneDownload() async throws {
        let data = makeData(1_000_000)
        let server = StubServer.serving(data)
        server.chunkDelay = .milliseconds(5)
        try await withDownloader(server: server) { downloader, _ in
            try await withThrowingTaskGroup(of: URL.self) { group in
                for _ in 0..<4 { group.addTask { try await downloader.ensure(source(data)) } }
                var urls = Set<URL>()
                for try await url in group { urls.insert(url) }
                #expect(urls.count == 1)
            }
            #expect(server.requests.count == 1)
        }
    }

    @Test func theStreamEndsWithTheFinishedFile() async throws {
        let data = makeData(300_000)
        try await withDownloader(server: .serving(data)) { downloader, models in
            var progress = 0
            var finished: URL?
            for try await event in downloader.download(source(data)) {
                switch event {
                case .progress: progress += 1
                case .finished(let url): finished = url
                }
            }
            #expect(progress > 0 && finished == models.appendingPathComponent("model.gguf"))
        }
    }

    @Test func removeDeletesTheModelItsPartAndItsMarker() async throws {
        let data = makeData(100_000)
        try await withDownloader(server: .serving(data)) { downloader, models in
            _ = try await downloader.ensure(source(data))
            try Data([1]).write(to: models.appendingPathComponent("model.gguf.part"))
            try await downloader.remove(source(data))
            #expect((try FileManager.default.contentsOfDirectory(atPath: models.path)).isEmpty)
            #expect(await downloader.localURL(for: source(data)) == nil)
        }
    }
}

@Suite("Model download sources")
struct ModelSourceValidationTests {
    @Test(arguments: [
        ModelSource.hub(repo: "no-slash", file: "m.gguf", sha256: nil),
        .hub(repo: "a/b/c", file: "m.gguf", sha256: nil),
        .hub(repo: "../evil", file: "m.gguf", sha256: nil),
        .hub(repo: "owner/name", file: "../escape.gguf", sha256: nil),
        .hub(repo: "owner/name", file: "dir/file.gguf", sha256: nil),
        .hub(repo: "owner/name", file: ".hidden", sha256: nil),
        .hub(repo: "owner/name", file: "", sha256: nil),
        .hub(repo: "owner/ na me", file: "m.gguf", sha256: nil),
        .file(URL(fileURLWithPath: "/tmp/x.gguf")),
    ])
    func malformedOrNonHubSourcesAreRejectedWithoutTouchingTheNetwork(source: ModelSource) async throws {
        let server = StubServer.serving(Data([1]))
        try await withDownloader(server: server) { downloader, models in
            await #expect(throws: ModelHubError.self) { _ = try await downloader.ensure(source) }
            #expect(server.requests.isEmpty)
            #expect(await downloader.localURL(for: source) == nil)
            #expect(!FileManager.default.fileExists(atPath: models.deletingLastPathComponent().appendingPathComponent("escape.gguf").path))
        }
    }

    @Test func theDownloadAddressIsBuiltAndEscapedCorrectly() throws {
        let url = try ModelDownloader.resolveURL(base: ModelDownloader.huggingFace, repo: "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF", file: "my model+v1.gguf")
        #expect(url.absoluteString == "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/my%20model+v1.gguf")
    }

    @Test func theDefaultFolderIsTheOneKnowledgeBaseSearches() throws {
        #expect(ModelResolver.modelsDirectory?.standardizedFileURL == ModelDownloader.defaultDirectory.standardizedFileURL)
    }

    @Test func aDownloadedHubModelIsFoundByKnowledgeBaseResolution() async throws {
        let data = makeData(20_000)
        let file = "resolver-test-\(UUID().uuidString.prefix(8)).gguf"
        let models = try #require(ModelResolver.modelsDirectory)
        defer { try? FileManager.default.removeItem(at: models.appendingPathComponent(file)) }
        let downloader = ModelDownloader(configuration: StubServer.serving(data).configuration, baseURL: URL(string: "https://never-called.test") ?? ModelDownloader.huggingFace)
        let hub = ModelSource.hub(repo: "owner/name", file: file, sha256: nil)
        #expect(throws: LociError.self) { try ModelResolver.url(for: hub) }
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        try data.write(to: models.appendingPathComponent(file))
        #expect(try ModelResolver.url(for: hub).lastPathComponent == file)
        #expect(await downloader.isDownloaded(hub), "an existing file without a checksum counts as present")
    }

    @Test func contentRangeParsing() {
        #expect(DownloadSession.start(of: "bytes 100-199/1000") == 100 && DownloadSession.total(of: "bytes 100-199/1000") == 1000)
        #expect(DownloadSession.total(of: "bytes */1000") == 1000 && DownloadSession.total(of: "bytes 0-9/*") == nil)
        #expect(DownloadSession.start(of: "garbage") == nil)
    }
}
