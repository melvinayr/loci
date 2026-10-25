import CoreGraphics
import CoreText
import Foundation
import LociLlama
import os

enum Fixtures {
    static let documents = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // LociTests
        .deletingLastPathComponent()  // Tests
        .appendingPathComponent("Fixtures/documents", isDirectory: true)

    static func url(_ name: String) -> URL { documents.appendingPathComponent(name) }
}

/// Makes a unique temporary directory that the caller removes.
func makeTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("loci-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

/// Writes a PDF with one entry of `pages` per page. A nil entry draws a shape and no text
/// (like a scanned page). With `password`, the PDF is encrypted.
func writePDF(pages: [String?], to url: URL, password: String? = nil) throws {
    var box = CGRect(x: 0, y: 0, width: 612, height: 792)
    var info: [CFString: Any] = [:]
    if let password {
        info[kCGPDFContextUserPassword] = password
        info[kCGPDFContextOwnerPassword] = password
    }
    guard let context = CGContext(url as CFURL, mediaBox: &box, info as CFDictionary) else {
        throw CocoaError(.fileWriteUnknown)
    }
    let font = CTFontCreateWithName("Helvetica" as CFString, 12, nil)
    for page in pages {
        context.beginPDFPage(nil)
        if let page {
            let text = NSAttributedString(string: page, attributes: [.font: font])
            let framesetter = CTFramesetterCreateWithAttributedString(text)
            let path = CGPath(rect: box.insetBy(dx: 54, dy: 54), transform: nil)
            CTFrameDraw(CTFramesetterCreateFrame(framesetter, CFRange(location: 0, length: 0), path, nil), context)
        } else {
            context.setFillColor(CGColor(gray: 0.5, alpha: 1))
            context.fill(CGRect(x: 100, y: 100, width: 200, height: 200))
        }
        context.endPDFPage()
    }
    context.closePDF()
}

extension String {
    /// Slice by Unicode-scalar offsets, the unit used throughout Loci.
    func scalarSlice(_ range: Range<Int>) -> String {
        let scalars = Array(unicodeScalars)
        return String(String.UnicodeScalarView(scalars[range]))
    }

    var scalarCount: Int { unicodeScalars.count }
}


/// A deterministic embedder for model-free tests: hashed bag-of-words, so texts that share
/// words are close and unrelated texts are not. Counts how many texts it has embedded.
final class FakeEmbedder: TextEmbedder {
    let identity: EmbedderIdentity
    private let embedded = OSAllocatedUnfairLock(initialState: 0)

    init(id: String = "fake|64|0", dimension: Int = 64) {
        identity = EmbedderIdentity(id: id, dimension: dimension)
    }

    var embeddedTextCount: Int { embedded.withLock { $0 } }

    func embed(_ texts: [String], role: EmbeddingRole) async throws -> [[Float]] {
        embedded.withLock { $0 += texts.count }
        return texts.map(vector)
    }

    func countTokens(_ scalars: ArraySlice<Unicode.Scalar>) -> Int {
        scalars.isEmpty ? 0 : (scalars.count + 3) / 4
    }

    private func vector(_ text: String) -> [Float] {
        var values = [Float](repeating: 0, count: identity.dimension)
        for word in text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }) {
            var hash: UInt64 = 0xcbf29ce484222325
            for byte in word.utf8 { hash = (hash ^ UInt64(byte)) &* 0x100000001b3 }
            values[Int(hash % UInt64(values.count))] += (hash >> 40) & 1 == 0 ? 1 : -1
        }
        let norm = values.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return norm > 0 ? values.map { $0 / norm } : values
    }
}


/// A deterministic reranker for model-free tests: scores a passage by how many of the query's
/// words it contains. Records what it was asked to score.
final class FakeReranker: TextReranker {
    let id: String
    private let scorer: @Sendable (String, String) -> Float
    private let log = OSAllocatedUnfairLock(initialState: (calls: 0, passages: [[String]](), queries: [String](), instructions: [String?]()))
    private let failureState = OSAllocatedUnfairLock<(any Error)?>(initialState: nil)
    var failure: (any Error)? {
        get { failureState.withLock { $0 } }
        set { failureState.withLock { $0 = newValue } }
    }

    init(id: String = "fake-reranker|1|0", scorer: (@Sendable (String, String) -> Float)? = nil) {
        self.id = id
        self.scorer = scorer ?? { query, passage in
            let words = Set(query.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init).filter { $0.count > 2 })
            let text = passage.lowercased()
            return Float(words.filter { text.contains($0) }.count)
        }
    }

    var callCount: Int { log.withLock { $0.calls } }
    var scoredPassages: [[String]] { log.withLock { $0.passages } }
    var queries: [String] { log.withLock { $0.queries } }
    var instructions: [String?] { log.withLock { $0.instructions } }

    func score(query: String, passages: [String], instruction: String?) async throws -> [Float] {
        log.withLock { $0.calls += 1; $0.passages.append(passages); $0.queries.append(query); $0.instructions.append(instruction) }
        if let failure { throw failure }
        return passages.map { scorer(query, $0) }
    }
}


/// A scripted generator for model-free tests: replies with fixed text in small pieces and records
/// what it was asked. Token counts are characters divided by four.
final class FakeGenerator: TextGenerator {
    let contextTokens: Int
    private let reply: @Sendable ([ChatMessage]) -> String
    private let pieceSize: Int
    private let delay: Duration?
    private let log = OSAllocatedUnfairLock(initialState: (conversations: [[ChatMessage]](), parameters: [GenerationParameters]()))
    private let failureState = OSAllocatedUnfairLock<(any Error)?>(initialState: nil)
    var failure: (any Error)? {
        get { failureState.withLock { $0 } }
        set { failureState.withLock { $0 = newValue } }
    }

    init(contextTokens: Int = 4096, pieceSize: Int = 7, delay: Duration? = nil, reply: @escaping @Sendable ([ChatMessage]) -> String) {
        self.contextTokens = contextTokens
        self.pieceSize = pieceSize
        self.delay = delay
        self.reply = reply
    }

    convenience init(contextTokens: Int = 4096, _ text: String) {
        self.init(contextTokens: contextTokens) { _ in text }
    }

    var conversations: [[ChatMessage]] { log.withLock { $0.conversations } }
    var parameters: [GenerationParameters] { log.withLock { $0.parameters } }

    func promptTokenCount(_ messages: [ChatMessage]) async throws -> Int {
        messages.reduce(0) { $0 + ($1.content.count + 3) / 4 + 4 }
    }

    func sanitize(_ text: String) -> String {
        text.replacingOccurrences(of: "<|im_end|>", with: "").replacingOccurrences(of: "<|im_start|>", with: "")
    }

    func generate(_ messages: [ChatMessage], parameters: GenerationParameters) -> AsyncThrowingStream<GenerationEvent, Error> {
        log.withLock { $0.conversations.append(messages); $0.parameters.append(parameters) }
        let text = reply(messages)
        let pieceSize = self.pieceSize, delay = self.delay, failure = self.failure
        return AsyncThrowingStream { continuation in
            let task = Task {
                if let failure { continuation.finish(throwing: failure); return }
                var pieces = 0
                var index = text.startIndex
                while index < text.endIndex {
                    if Task.isCancelled { continuation.finish(throwing: CancellationError()); return }
                    if let delay { try? await Task.sleep(for: delay) }
                    let end = text.index(index, offsetBy: pieceSize, limitedBy: text.endIndex) ?? text.endIndex
                    continuation.yield(.text(String(text[index..<end])))
                    pieces += 1
                    index = end
                }
                continuation.yield(.finished(GenerationStats(
                    promptTokens: messages.reduce(0) { $0 + $1.content.count / 4 }, generatedTokens: pieces,
                    timeToFirstToken: .milliseconds(5), tokensPerSecond: 42, stopReason: .endOfText)))
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
