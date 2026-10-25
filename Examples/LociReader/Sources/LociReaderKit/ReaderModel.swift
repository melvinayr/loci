import Foundation
import Loci
import SwiftUI

/// Where the cited passage is shown.
public enum SourceContent {
    /// A PDF file, to be opened at `page` with `excerpt` marked.
    case pdf(url: URL, page: Int, excerpt: String)
    /// A text-like document with the passage marked.
    case text(title: String, HighlightedText)
    /// The document's file is gone; the excerpt is all there is to show.
    case excerptOnly(title: String, page: Int?, excerpt: String)
}

/// The state of the reader screen: library, question, streamed answer and the source being inspected.
@MainActor
public final class ReaderModel: ObservableObject {
    public enum ImportState: Equatable {
        case idle
        case importing(current: String, completed: Int, total: Int)
        /// Finished; `problems` has one line per file that could not be added.
        case finished(added: Int, problems: [String])
    }

    public enum Phase: Equatable {
        case idle
        case searching
        case writing
        case checking
        case finished
        case failed(String)

        public var isBusy: Bool {
            switch self {
            case .searching, .writing, .checking: true
            case .idle, .finished, .failed: false
            }
        }
    }

    @Published public private(set) var documents: [Loci.Document] = []
    @Published public private(set) var importState: ImportState = .idle
    @Published public var question = ""
    @Published public private(set) var phase: Phase = .idle
    /// The passages the answer draws on, as soon as they are known.
    @Published public private(set) var sources: [SearchResult] = []
    /// The reply as it is generated, with the model's own citation numbers.
    @Published public private(set) var draft = ""
    /// Sentences as they are checked.
    @Published public private(set) var sentences: [AnsweredSentence] = []
    @Published public private(set) var answer: Answer?
    @Published public private(set) var selectedCitationIndex: Int?
    @Published public private(set) var source: SourceContent?

    private let engine: any ReaderEngine
    private var answerTask: Task<Void, Never>?

    public init(engine: any ReaderEngine) {
        self.engine = engine
    }

    // MARK: Library

    public func refresh() async {
        documents = (try? await engine.documents()) ?? documents
    }

    /// Adds the files, continuing past ones that fail and reporting why.
    public func importFiles(_ urls: [URL]) async {
        var added = 0
        var problems: [String] = []
        for (index, url) in urls.enumerated() {
            importState = .importing(current: url.lastPathComponent, completed: index, total: urls.count)
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            do {
                _ = try await engine.add(fileAt: url)
                added += 1
            } catch {
                problems.append("\(url.lastPathComponent): \(error.localizedDescription)")
            }
        }
        importState = .finished(added: added, problems: problems)
        await refresh()
    }

    public func remove(_ document: Loci.Document) async {
        try? await engine.remove(document.id)
        if case .pdf = source { clearSource() }
        await refresh()
    }

    // MARK: Asking

    public var canAsk: Bool { !phase.isBusy && !question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !documents.isEmpty }

    public func ask() {
        guard canAsk else { return }
        let text = question.trimmingCharacters(in: .whitespacesAndNewlines)
        resetAnswer()
        phase = .searching
        answerTask = Task { [weak self, engine] in
            do {
                for try await event in engine.answer(to: text) {
                    self?.handle(event)
                }
            } catch is CancellationError {
                self?.phase = .idle
            } catch {
                self?.phase = .failed(error.localizedDescription)
            }
        }
    }

    /// Stops the current answer, keeping nothing of it.
    public func cancel() {
        answerTask?.cancel()
        answerTask = nil
        if phase.isBusy { phase = .idle }
    }

    private func resetAnswer() {
        sources = []
        draft = ""
        sentences = []
        answer = nil
        selectedCitationIndex = nil
        source = nil
    }

    private func handle(_ event: AnswerEvent) {
        switch event {
        case .retrieved(let results):
            sources = results
            phase = .writing
        case .token(let piece):
            draft += piece
            phase = .writing
        case .sentence(let sentence):
            sentences.append(sentence)
            phase = .checking
        case .finished(let finished):
            answer = finished
            sentences = finished.sentences
            phase = .finished
        }
    }

    // MARK: Sources

    /// Opens the passage behind citation `index` of the current answer.
    public func showCitation(_ index: Int) async {
        guard let citation = answer?.citations.first(where: { $0.index == index }) else { return }
        selectedCitationIndex = index
        source = await content(document: citation.document, page: citation.page, range: citation.range, excerpt: citation.excerpt)
    }

    /// Opens a retrieved passage (for a refusal, which cites nothing but lists what was looked at).
    public func showSource(_ result: SearchResult) async {
        selectedCitationIndex = nil
        source = await content(document: result.document, page: result.page, range: result.range, excerpt: result.text)
    }

    public func clearSource() {
        source = nil
        selectedCitationIndex = nil
    }

    private func content(document: Loci.Document, page: Int?, range: Range<Int>, excerpt: String) async -> SourceContent {
        if document.kind == .pdf, let url = engine.fileURL(of: document) {
            return .pdf(url: url, page: page ?? 1, excerpt: excerpt)
        }
        if document.kind != .pdf, let text = try? await engine.text(of: document.id) {
            return .text(title: document.title, HighlightedText.make(documentText: text, scalarRange: range))
        }
        return .excerptOnly(title: document.title, page: page, excerpt: excerpt)
    }

    // MARK: Presentation helpers

    /// A line summarising how well the answer is supported.
    public var supportSummary: String? {
        guard let support = answer?.support, let answer, !answer.sentences.isEmpty else { return nil }
        let supported = answer.sentences.filter { $0.support == .supported }.count
        return "\(supported) of \(answer.sentences.count) sentences supported by their sources"
            + (support.level == .supported ? "" : " (check the flagged ones)")
    }
}
