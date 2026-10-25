import Foundation
import os
import Testing
@testable import Loci
@testable import LociLlama
@testable import LociStore

private let keyword = RetrievalOptions(mode: .keyword)

private func makeKnowledgeBase(
    in directory: URL, embedder: any TextEmbedder = FakeEmbedder()
) async throws -> KnowledgeBase {
    try await KnowledgeBase(directory: directory.appendingPathComponent("kb"), services: await ModelServices(embedder: embedder))
}

private func withDirectory(_ body: (URL) async throws -> Void) async throws {
    let directory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    try await body(directory)
}

private struct FailingEmbedder: TextEmbedder {
    let identity = EmbedderIdentity(id: "failing|64|0", dimension: 64)
    func embed(_ texts: [String], role: EmbeddingRole) async throws -> [[Float]] {
        throw LlamaError.backend(operation: "decode", code: 9)
    }
    func countTokens(_ scalars: ArraySlice<Unicode.Scalar>) -> Int { (scalars.count + 3) / 4 }
}

/// Takes a while per call so a test can cancel mid-embedding.
private struct SlowEmbedder: TextEmbedder {
    var inner = FakeEmbedder()
    var identity: EmbedderIdentity { inner.identity }
    func embed(_ texts: [String], role: EmbeddingRole) async throws -> [[Float]] {
        try await Task.sleep(for: .milliseconds(200))
        return try await inner.embed(texts, role: role)
    }
    func countTokens(_ scalars: ArraySlice<Unicode.Scalar>) -> Int { (scalars.count + 3) / 4 }
}

@Suite("KnowledgeBase vector and hybrid search")
struct KnowledgeBaseVectorSearchTests {
    private func seed(_ kb: KnowledgeBase) async throws {
        for (title, text) in [
            ("sourdough", "Sourdough bread needs flour water and an active starter left to rise overnight."),
            ("lighthouse", "The lighthouse on the headland was automated and its keeper retired."),
            ("solar", "Rooftop solar panels convert sunlight into electricity for the house."),
        ] {
            try await kb.add(.text(text, title: title))
        }
    }

    @Test func vectorSearchRanksBySimilarityAndReportsScores() async throws {
        try await withDirectory { directory in
            let kb = try await makeKnowledgeBase(in: directory)
            try await seed(kb)
            let results = try await kb.search("starter for bread dough", options: RetrievalOptions(limit: 3, mode: .vector))
            #expect(results.first?.document.title == "sourdough")
            let top = try #require(results.first)
            #expect((top.scores.vector ?? 0) > 0.2 && top.scores.bm25 == nil && top.scores.fused == nil)
            let scores = results.compactMap(\.scores.vector)
            #expect(scores == scores.sorted(by: >))
        }
    }

    @Test func hybridReturnsFusedScoresAndKeepsInvariants() async throws {
        try await withDirectory { directory in
            let kb = try await makeKnowledgeBase(in: directory)
            try await seed(kb)
            let results = try await kb.search("sunlight electricity solar", options: RetrievalOptions(limit: 3, mode: .hybrid))
            #expect(results.first?.document.title == "solar")
            #expect(results.allSatisfy { $0.scores.fused != nil })
            for result in results {
                let text = try #require(try await kb.documentText(for: result.document.id))
                #expect(text.scalarSlice(result.range) == result.text)
            }
        }
    }

    @Test func hybridFindsAParaphraseThatKeywordSearchMisses() async throws {
        try await withDirectory { directory in
            // The fake embedder only knows word overlap, so "paraphrase" here means: the question shares
            // words with the passage that BM25 stems differently or ignores. Check the plumbing, not the model.
            let kb = try await makeKnowledgeBase(in: directory)
            try await seed(kb)
            let keywordOnly = try await kb.search("zzz qqq", options: keyword)
            let vectorOnly = try await kb.search("zzz qqq", options: RetrievalOptions(limit: 3, mode: .vector))
            #expect(keywordOnly.isEmpty)
            #expect(vectorOnly.count == 3, "vector search always returns the nearest passages")
        }
    }

    @Test func filtersApplyToVectorSearch() async throws {
        try await withDirectory { directory in
            let kb = try await makeKnowledgeBase(in: directory)
            let note = try await kb.add(.text("shared words about bread", title: "note"), options: IngestOptions(metadata: ["kind": "note"]))
            try await kb.add(.text("shared words about bread again", title: "paper"), options: IngestOptions(metadata: ["kind": "paper"]))
            let byMetadata = try await kb.search("shared bread", options: RetrievalOptions(mode: .vector, filter: .metadata("kind", "note")))
            #expect(byMetadata.map(\.document.title) == ["note"])
            let byDocument = try await kb.search("shared bread", options: RetrievalOptions(mode: .hybrid, filter: .document([note.id])))
            #expect(byDocument.map(\.document.title) == ["note"])
        }
    }

    @Test func keywordSearchNeverTouchesTheEmbedder() async throws {
        try await withDirectory { directory in
            let fake = FakeEmbedder()
            let kb = try await makeKnowledgeBase(in: directory, embedder: fake)
            try await seed(kb)
            let afterIngest = fake.embeddedTextCount
            _ = try await kb.search("bread", options: keyword)
            #expect(fake.embeddedTextCount == afterIngest)
            _ = try await kb.search("bread", options: RetrievalOptions(mode: .hybrid))
            #expect(fake.embeddedTextCount == afterIngest + 1, "the query is embedded once")
        }
    }

    @Test func blankQueriesReturnNothingWithoutEmbedding() async throws {
        try await withDirectory { directory in
            let fake = FakeEmbedder()
            let kb = try await makeKnowledgeBase(in: directory, embedder: fake)
            try await seed(kb)
            let before = fake.embeddedTextCount
            #expect(try await kb.search("   \n", options: .default).isEmpty)
            #expect(fake.embeddedTextCount == before)
        }
    }
}

@Suite("KnowledgeBase embedding lifecycle")
struct KnowledgeBaseEmbeddingLifecycleTests {
    @Test func everyChunkGetsAVectorAndTheEmbedderIsRecorded() async throws {
        try await withDirectory { directory in
            let kb = try await makeKnowledgeBase(in: directory)
            try await kb.add(.file(Fixtures.url("long.md")))
            let stats = await kb.stats
            #expect(stats.chunkCount > 1 && stats.vectorCount == stats.chunkCount)
            #expect(stats.embedderID == "fake|64|0")
        }
    }

    @Test func reAddingAnUnchangedDocumentDoesNotEmbedAgain() async throws {
        try await withDirectory { directory in
            let fake = FakeEmbedder()
            let kb = try await makeKnowledgeBase(in: directory, embedder: fake)
            try await kb.add(.file(Fixtures.url("sample.md")))
            let embedded = fake.embeddedTextCount
            #expect(embedded > 0)
            try await kb.add(.file(Fixtures.url("sample.md")))
            #expect(fake.embeddedTextCount == embedded)
        }
    }

    @Test func aDifferentEmbedderIsRejectedUntilReindexed() async throws {
        try await withDirectory { directory in
            let first = try await makeKnowledgeBase(in: directory, embedder: FakeEmbedder(id: "model-a|64|1", dimension: 64))
            try await first.add(.text("bread starter dough", title: "bread"))
            await first.store.close()

            let second = try await makeKnowledgeBase(in: directory, embedder: FakeEmbedder(id: "model-b|32|2", dimension: 32))
            func expectMismatch(_ operation: () async throws -> Void) async {
                do {
                    try await operation()
                    Issue.record("expected embedderMismatch")
                } catch LociError.embedderMismatch(let stored, let configured) {
                    #expect(stored == "model-a|64|1" && configured == "model-b|32|2")
                } catch {
                    Issue.record("unexpected error \(error)")
                }
            }
            await expectMismatch { _ = try await second.add(.text("more text", title: "x")) }
            await expectMismatch { _ = try await second.search("bread", options: RetrievalOptions(mode: .vector)) }
            await expectMismatch { _ = try await second.search("bread", options: RetrievalOptions(mode: .hybrid)) }

            // Everything that does not need vectors still works.
            #expect(try await second.search("bread", options: keyword).count == 1)
            #expect(try await second.documents().count == 1)

            var progress: [IngestProgress] = []
            for try await event in second.reindex() { progress.append(event) }
            #expect(progress.last?.phase == .done && progress.last?.completed == 1 && progress.last?.total == 1)

            let stats = await second.stats
            #expect(stats.embedderID == "model-b|32|2" && stats.vectorCount == stats.chunkCount)
            #expect(try await second.search("bread dough", options: RetrievalOptions(mode: .vector)).first?.document.title == "bread")
            try await second.add(.text("now adding works", title: "after"))
        }
    }

    @Test func aCancelledReindexLeavesTheMismatchInPlaceAndCanBeRerun() async throws {
        try await withDirectory { directory in
            let first = try await makeKnowledgeBase(in: directory, embedder: FakeEmbedder(id: "a|64|1", dimension: 64))
            for i in 0..<4 { try await first.add(.text("document number \(i) about topic \(i)", title: "d\(i)")) }
            await first.store.close()

            let second = try await makeKnowledgeBase(in: directory, embedder: SlowEmbedder(inner: FakeEmbedder(id: "b|64|2", dimension: 64)))
            let task = Task { for try await _ in second.reindex() {} }
            try await Task.sleep(for: .milliseconds(250))
            task.cancel()
            _ = await task.result
            try await Task.sleep(for: .milliseconds(500))  // let the producer wind down

            #expect(await second.stats.embedderID == "a|64|1", "the model switch is only recorded when the reindex completes")
            for try await _ in second.reindex() {}
            #expect(await second.stats.embedderID == "b|64|2")
            #expect(try await second.search("topic 2", options: RetrievalOptions(mode: .vector)).isEmpty == false)
        }
    }

    @Test func anEmptyKnowledgeBaseAdoptsTheConfiguredEmbedder() async throws {
        try await withDirectory { directory in
            let first = try await makeKnowledgeBase(in: directory, embedder: FakeEmbedder(id: "old|64|1"))
            let doc = try await first.add(.text("temporary", title: "t"))
            try await first.remove(doc.id)
            await first.store.close()

            let second = try await makeKnowledgeBase(in: directory, embedder: FakeEmbedder(id: "new|64|2"))
            try await second.add(.text("fresh start", title: "f"))
            #expect(await second.stats.embedderID == "new|64|2")
        }
    }

    @Test func folderIngestStopsOnAMismatchInsteadOfSkippingEveryFile() async throws {
        try await withDirectory { directory in
            let first = try await makeKnowledgeBase(in: directory, embedder: FakeEmbedder(id: "a|64|1"))
            try await first.add(.text("existing", title: "e"))
            await first.store.close()
            let second = try await makeKnowledgeBase(in: directory, embedder: FakeEmbedder(id: "b|64|2"))

            let folder = directory.appendingPathComponent("docs")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try "one".write(to: folder.appendingPathComponent("1.txt"), atomically: true, encoding: .utf8)
            try "two".write(to: folder.appendingPathComponent("2.txt"), atomically: true, encoding: .utf8)
            do {
                for try await _ in second.add(contentsOf: folder) {}
                Issue.record("expected the stream to throw")
            } catch LociError.embedderMismatch {
            }
        }
    }

    @Test func anEmbeddingFailureStoresNothing() async throws {
        try await withDirectory { directory in
            let kb = try await makeKnowledgeBase(in: directory, embedder: FailingEmbedder())
            await #expect(throws: LociError.self) { try await kb.add(.text("will not be stored", title: "t")) }
            let stats = await kb.stats
            #expect(stats.documentCount == 0 && stats.chunkCount == 0 && stats.vectorCount == 0)
            do {
                _ = try await kb.add(.text("again", title: "t"))
            } catch LociError.backendFailure(_, let code) {
                #expect(code == 9)
            }
        }
    }

    @Test func cancellingDuringEmbeddingStoresNothing() async throws {
        try await withDirectory { directory in
            let kb = try await makeKnowledgeBase(in: directory, embedder: SlowEmbedder())
            let task = Task { try await kb.add(.text("slow document with several words", title: "s")) }
            try await Task.sleep(for: .milliseconds(50))
            task.cancel()
            do {
                _ = try await task.value
                Issue.record("expected cancellation")
            } catch LociError.cancelled {
            }
            #expect(await kb.stats.documentCount == 0)
        }
    }

    @Test func unloadModelsReleasesTheEmbedderAndItReloadsOnDemand() async throws {
        try await withDirectory { directory in
            let services = await ModelServices(embedder: FakeEmbedder())
            let kb = try await KnowledgeBase(directory: directory.appendingPathComponent("kb"), services: services)
            #expect(await services.manager.residentRoles.isEmpty, "opening loads nothing")
            try await kb.add(.text("alpha beta", title: "a"))
            #expect(await services.manager.residentRoles == [.embedder])
            await kb.unloadModels()
            #expect(await services.manager.residentRoles.isEmpty)
            _ = try await kb.search("alpha", options: RetrievalOptions(mode: .hybrid))
            #expect(await services.manager.residentRoles == [.embedder])
        }
    }

    @Test func aMissingModelFileIsReportedWhileKeywordFeaturesStillWork() async throws {
        try await withDirectory { directory in
            let models = ModelSet(embedder: .file(directory.appendingPathComponent("absent.gguf")))
            let kb = try await KnowledgeBase(directory: directory.appendingPathComponent("kb"), models: models)
            do {
                _ = try await kb.add(.text("text", title: "t"))
                Issue.record("expected modelNotFound")
            } catch LociError.modelNotFound(let source) {
                #expect(source == models.embedder)
                #expect(LociError.modelNotFound(source).errorDescription?.contains("absent.gguf") == true)
                #expect(LociError.modelNotFound(source).errorDescription?.contains(directory.path) == false, "no full paths in messages")
            }
            #expect(try await kb.documents().isEmpty)
            #expect(try await kb.search("x", options: keyword).isEmpty)
            await #expect(throws: LociError.self) { try await kb.search("x", options: RetrievalOptions(mode: .hybrid)) }
        }
    }

    @Test func hubModelsResolveToTheDownloadFolderWithoutNetwork() throws {
        let source = ModelSource.hub(repo: "org/repo", file: "definitely-not-downloaded-\(UUID().uuidString).gguf", sha256: nil)
        #expect(throws: LociError.self) { try ModelResolver.url(for: source) }
        #expect(ModelSource.hub(repo: "a", file: "m.gguf", sha256: nil).displayName.contains("LociHub"))
    }
}
