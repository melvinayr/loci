import Foundation
import Loci
import PDFKit
import Testing

@testable import LociReaderKit

private let root = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
private let embedder = root.appendingPathComponent("Tests/Fixtures/models/bge-small-en-v1.5-q8_0.gguf")
private let reranker = root.appendingPathComponent("Tests/Fixtures/models/qwen3-reranker-0.6b-q8_0.gguf")
private let generator = root.appendingPathComponent("Tests/Fixtures/models/qwen3-0.6b-q8_0.gguf")
private let documents = root.appendingPathComponent("Tests/Fixtures/eval-small/documents")
private let present = [embedder, reranker, generator].allSatisfy { FileManager.default.fileExists(atPath: $0.path) }

@MainActor
private func makeModel(in directory: URL) async throws -> ReaderModel {
    let kb = try await KnowledgeBase(
        directory: directory.appendingPathComponent("kb"),
        models: ModelSet(embedder: .file(embedder), reranker: .file(reranker), generator: .file(generator)),
        options: KnowledgeBaseOptions(residency: .keepResident, gpuLayers: .none))
    let engine = KnowledgeBaseEngine(
        knowledgeBase: kb, library: directory.appendingPathComponent("library"),
        answerOptions: AnswerOptions(maxAnswerTokens: 120, temperature: 0))
    return ReaderModel(engine: engine)
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("loci-reader-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

@Suite("Reader model with the fixture models", .serialized, .enabled(if: present))
@MainActor
struct ReaderModelTests {
    @Test func importsFilesAndReportsFailures() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try await makeModel(in: directory)
        let good = documents.appendingPathComponent("cape-marrow-lighthouse.md")
        let missing = directory.appendingPathComponent("missing.pdf")
        await model.importFiles([good, missing])
        #expect(model.documents.count == 1)
        guard case .finished(let added, let problems) = model.importState else {
            Issue.record("import did not finish")
            return
        }
        #expect(added == 1)
        #expect(problems.count == 1)
        #expect(problems[0].hasPrefix("missing.pdf"))
    }

    @Test func removingADocumentDeletesItsCopy() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try await makeModel(in: directory)
        await model.importFiles([documents.appendingPathComponent("sourdough-guide.txt")])
        let library = directory.appendingPathComponent("library")
        #expect(try FileManager.default.contentsOfDirectory(atPath: library.path).count == 1)
        let document = try #require(model.documents.first)
        await model.remove(document)
        #expect(model.documents.isEmpty)
        #expect(try FileManager.default.contentsOfDirectory(atPath: library.path).isEmpty)
    }

    @Test func answerStreamsAndCitationsOpenTheirSource() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try await makeModel(in: directory)
        await model.importFiles([
            documents.appendingPathComponent("cape-marrow-lighthouse.md"),
            documents.appendingPathComponent("brindlemoor-trial-report.pdf"),
        ])
        #expect(model.documents.count == 2)

        model.question = "How many participants did the Brindlemoor trial enrol?"
        #expect(model.canAsk)
        model.ask()
        while model.phase.isBusy { try await Task.sleep(nanoseconds: 50_000_000) }
        #expect(model.phase == .finished)
        let answer = try #require(model.answer)
        #expect(!model.sources.isEmpty)
        if answer.refused { return }  // nothing to cite

        let citation = try #require(answer.citations.first)
        await model.showCitation(citation.index)
        #expect(model.selectedCitationIndex == citation.index)
        switch try #require(model.source) {
        case .pdf(let url, let page, let excerpt):
            #expect(page == citation.page)
            #expect(excerpt == citation.excerpt)
            let document = try #require(PDFDocument(url: url))
            let target = try #require(PDFHighlighter.target(in: document, page: page, excerpt: excerpt))
            #expect(target.selection != nil)
        case .text(_, let highlighted):
            #expect(highlighted.highlighted == citation.excerpt)
        case .excerptOnly:
            Issue.record("the file should still be available")
        }
    }

    @Test func cancelStopsAndAskNeedsQuestionAndDocuments() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try await makeModel(in: directory)
        model.question = "anything"
        #expect(!model.canAsk)  // no documents yet
        await model.importFiles([documents.appendingPathComponent("sourdough-guide.txt")])
        model.question = "   "
        #expect(!model.canAsk)
        model.question = "How hot should the oven be?"
        model.ask()
        model.cancel()
        try await Task.sleep(nanoseconds: 300_000_000)
        #expect(!model.phase.isBusy)
    }
}

@Suite struct AnswerRenderingTests {
    @Test func markersBecomeLinks() {
        let text = AnswerRendering.attributed("Built in 1874 [1], electrified later [2][3].")
        let links = text.runs.compactMap(\.link)
        #expect(links.compactMap(AnswerRendering.citationIndex(from:)) == [1, 2, 3])
        #expect(String(text.characters) == "Built in 1874 [1], electrified later [2][3].")
    }

    @Test func foreignLinksAreNotCitations() throws {
        #expect(AnswerRendering.citationIndex(from: try #require(URL(string: "https://example.com"))) == nil)
    }

    @Test func plainTextIsUnchanged() {
        #expect(String(AnswerRendering.attributed("No markers [x] here.").characters) == "No markers [x] here.")
    }
}
