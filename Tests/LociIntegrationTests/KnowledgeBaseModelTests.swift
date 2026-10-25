import Foundation
import Testing
import Loci
import LociEval

private let evalSmall = ModelFixtures.repositoryRoot.appendingPathComponent("Tests/Fixtures/eval-small", isDirectory: true)

private func models(_ embedder: URL) -> ModelSet { ModelSet(embedder: .file(embedder)) }

private func withTemporaryDirectory(_ body: (URL) async throws -> Void) async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("loci-it-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try await body(directory)
}

/// End-to-end with real GGUF embedders on CPU: ingest → embed → hybrid search.
@Suite("KnowledgeBase with real embedders", .enabled(if: ModelFixtures.exists(ModelFixtures.fixtureEmbedder) || ModelFixtures.exists(ModelFixtures.localEmbedder)))
struct KnowledgeBaseModelTests {
    private let embedder = ModelFixtures.embedder

    private func openKnowledgeBase(_ directory: URL, embedder: URL? = nil) async throws -> KnowledgeBase {
        try await KnowledgeBase(
            directory: directory.appendingPathComponent("kb"), models: models(embedder ?? self.embedder),
            options: KnowledgeBaseOptions(gpuLayers: .none))
    }

    @Test func hybridSearchAnswersParaphrasedQuestionsFromTheRightDocument() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory)
            for try await _ in kb.add(contentsOf: evalSmall.appendingPathComponent("documents"), options: IngestOptions(targetChunkTokens: 64, overlapTokens: 8)) {}

            let stats = await kb.stats
            #expect(stats.documentCount == 8 && stats.vectorCount == stats.chunkCount && stats.chunkCount > 40)

            let cases: [(question: String, document: String)] = [
                ("How long does an inverter last?", "home-solar-install.html"),
                ("Which people were not allowed to take part in the study?", "brindlemoor-trial-report.pdf"),
                ("What happened to the guildhall in 1471?", "saltmoor-weavers-guild.html"),
                ("Why does my loaf come out flat?", "sourdough-guide.txt"),
            ]
            for (question, expected) in cases {
                let results = try await kb.search(question, options: RetrievalOptions(limit: 3, mode: .hybrid))
                let names = results.map { URL(string: $0.document.uri ?? "")?.lastPathComponent }
                #expect(names.first == expected, "“\(question)” → \(names)")
                #expect(results.allSatisfy { $0.scores.fused != nil })
            }
        }
    }

    @Test func citationInvariantsHoldWithRealTokenizerChunking() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory)
            let pdf = evalSmall.appendingPathComponent("documents/brindlemoor-trial-report.pdf")
            let document = try await kb.add(.file(pdf), options: IngestOptions(targetChunkTokens: 48, overlapTokens: 8))
            #expect(document.pageCount == 4 && document.chunkCount > 4)

            for query in ["dropout rate", "who funded the trial", "registration number", "nausea"] {
                for mode in [RetrievalMode.keyword, .vector, .hybrid] {
                    for result in try await kb.search(query, options: RetrievalOptions(limit: 5, mode: mode)) {
                        #expect(result.document.kind == .pdf)
                        let page = try #require(result.page)
                        #expect((1...4).contains(page))
                        #expect(!result.text.isEmpty && result.range.count > 0)
                    }
                }
            }
        }
    }

    @Test func evaluationOnTheFixtureDatasetMeetsAFloor() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory)
            let report = try await Evaluator(knowledgeBase: kb).run(dataset: .folder(evalSmall))
            for mode in ["keyword", "vector", "hybrid"] {
                let metrics = try #require(report.result(for: mode)).metrics
                #expect(metrics.recallAt5 >= 0.9, "\(mode) Recall@5 = \(metrics.recallAt5)")
                #expect(metrics.mrr >= 0.8, "\(mode) MRR = \(metrics.mrr)")
            }
        }
    }

    @Test func unloadingModelsAndReopeningKeepsSearchWorking() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory)
            try await kb.add(.text("The quokka is a small marsupial found on Rottnest Island.", title: "quokka"))
            await kb.unloadModels()
            let again = try await openKnowledgeBase(directory)
            let results = try await again.search("which small marsupial lives on Rottnest?", options: RetrievalOptions(mode: .vector))
            #expect(results.first?.document.title == "quokka")
        }
    }
}

@Suite(
    "Switching embedding models",
    .enabled(if: ModelFixtures.exists(ModelFixtures.fixtureEmbedder) && ModelFixtures.exists(ModelFixtures.localEmbedder)))
struct EmbedderSwitchTests {
    @Test func openingWithAnotherRealEmbedderNeedsAReindex() async throws {
        try await withTemporaryDirectory { directory in
            let location = directory.appendingPathComponent("kb")
            let options = KnowledgeBaseOptions(gpuLayers: .none)
            let first = try await KnowledgeBase(directory: location, models: models(ModelFixtures.fixtureEmbedder), options: options)
            try await first.add(.text("The quokka is a small marsupial found on Rottnest Island.", title: "quokka"))
            let before = await first.stats

            let second = try await KnowledgeBase(directory: location, models: models(ModelFixtures.localEmbedder), options: options)
            do {
                _ = try await second.search("marsupial", options: RetrievalOptions(mode: .vector))
                Issue.record("expected embedderMismatch")
            } catch LociError.embedderMismatch(let stored, let configured) {
                #expect(stored == before.embedderID && stored != configured)
            }
            #expect(try await second.search("marsupial", options: RetrievalOptions(mode: .keyword)).count == 1)

            for try await _ in second.reindex() {}
            let after = await second.stats
            #expect(after.embedderID != before.embedderID && after.vectorCount == after.chunkCount)
            #expect(try await second.search("small marsupial", options: RetrievalOptions(mode: .vector)).first?.document.title == "quokka")
        }
    }
}

@Suite("Model errors with real files")
struct ModelErrorTests {
    @Test func aNonEmbeddingFileIsReportedWithTheRoleNotAGenericFailure() async throws {
        try await withTemporaryDirectory { directory in
            let junk = directory.appendingPathComponent("not-a-model.gguf")
            try Data(repeating: 1, count: 2048).write(to: junk)
            let kb = try await KnowledgeBase(directory: directory.appendingPathComponent("kb"), models: models(junk))
            do {
                _ = try await kb.add(.text("text", title: "t"))
                Issue.record("expected an error")
            } catch LociError.invalidModel(let source, _) {
                #expect(source == .file(junk))
            }
        }
    }

    @Test(.enabled(if: ModelFixtures.exists(ModelFixtures.generator)))
    func aChatModelAsEmbedderIsRejectedAsARoleMismatch() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await KnowledgeBase(
                directory: directory.appendingPathComponent("kb"), models: models(ModelFixtures.generator),
                options: KnowledgeBaseOptions(gpuLayers: .none))
            do {
                _ = try await kb.add(.text("text", title: "t"))
                Issue.record("expected an error")
            } catch LociError.modelRoleMismatch(_, let expected) {
                #expect(expected == "embedding")
            }
        }
    }
}
