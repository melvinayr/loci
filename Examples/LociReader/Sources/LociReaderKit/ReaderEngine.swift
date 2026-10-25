import Foundation
import Loci

/// What the reader needs from a knowledge base. `KnowledgeBaseEngine` is the real one; tests use a scripted one.
public protocol ReaderEngine: Sendable {
    func documents() async throws -> [Document]
    /// Adds a copy of the file at `url` to the library and the knowledge base.
    func add(fileAt url: URL) async throws -> Document
    func remove(_ id: DocumentID) async throws
    func answer(to question: String) -> AsyncThrowingStream<AnswerEvent, Error>
    /// The normalised text citations of `id` refer to.
    func text(of id: DocumentID) async throws -> String?
    /// The file a document was imported from, if it still exists.
    func fileURL(of document: Document) -> URL?
}

/// The reader on a real `KnowledgeBase`. Imported files are copied into `library` so they stay readable after the
/// original is moved and, on iOS, outside the security scope of the picker.
public struct KnowledgeBaseEngine: ReaderEngine {
    public let knowledgeBase: KnowledgeBase
    public let library: URL
    public let answerOptions: AnswerOptions

    public init(knowledgeBase: KnowledgeBase, library: URL, answerOptions: AnswerOptions = AnswerOptions()) {
        self.knowledgeBase = knowledgeBase
        self.library = library
        self.answerOptions = answerOptions
    }

    public func documents() async throws -> [Document] {
        try await knowledgeBase.documents()
    }

    public func add(fileAt url: URL) async throws -> Document {
        let manager = FileManager.default
        try manager.createDirectory(at: library, withIntermediateDirectories: true)
        // A folder per import keeps equal file names from different places apart.
        let folder = library.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try manager.createDirectory(at: folder, withIntermediateDirectories: true)
        let copy = folder.appendingPathComponent(url.lastPathComponent)
        try manager.copyItem(at: url, to: copy)
        do {
            return try await knowledgeBase.add(.file(copy))
        } catch {
            try? manager.removeItem(at: folder)  // an unreadable file leaves no copy behind
            throw error
        }
    }

    public func remove(_ id: DocumentID) async throws {
        let document = try await knowledgeBase.documents().first { $0.id == id }
        try await knowledgeBase.remove(id)
        if let document, let url = fileURL(of: document), url.path.hasPrefix(library.path) {
            try? FileManager.default.removeItem(at: url.deletingLastPathComponent())
        }
    }

    public func answer(to question: String) -> AsyncThrowingStream<AnswerEvent, Error> {
        knowledgeBase.streamAnswer(question, options: answerOptions)
    }

    public func text(of id: DocumentID) async throws -> String? {
        try await knowledgeBase.documentText(for: id)
    }

    public func fileURL(of document: Document) -> URL? {
        guard let uri = document.uri, let url = URL(string: uri), url.isFileURL, FileManager.default.fileExists(atPath: url.path) else { return nil }
        return url
    }
}
