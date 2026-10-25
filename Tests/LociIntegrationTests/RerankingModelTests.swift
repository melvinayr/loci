import Foundation
import Testing
import Loci
@testable import LociLlama

private let modelsAvailable = ModelFixtures.exists(ModelFixtures.fixtureEmbedder) && ModelFixtures.exists(ModelFixtures.fixtureReranker)

private func withTemporaryDirectory(_ body: (URL) async throws -> Void) async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("loci-it-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try await body(directory)
}

private func openKnowledgeBase(_ directory: URL, residency: ModelResidency) async throws -> KnowledgeBase {
    try await KnowledgeBase(
        directory: directory.appendingPathComponent("kb"),
        models: ModelSet(embedder: .file(ModelFixtures.fixtureEmbedder), reranker: .file(ModelFixtures.fixtureReranker)),
        options: KnowledgeBaseOptions(residency: residency, gpuLayers: .none))
}

private let corpus: [(title: String, text: String)] = [
    ("trial", "In total 374 participants completed the treatment period, a dropout rate of 9.2 percent across the three arms. Nausea was the most common side effect."),
    ("lighthouse", "The Cape Marrow lighthouse was built in 1874 and its granite tower is 38 metres tall. The keeper rescued survivors from the wreck."),
    ("bread", "Feed the sourdough starter at a ratio of 1:1:1 by weight and bake the loaf at 245 degrees Celsius in a Dutch oven."),
]

/// A real embedder and a real cross-encoder end to end on CPU, with a small corpus so reranking stays quick.
@Suite("Reranking with real models", .serialized, .enabled(if: modelsAvailable))
struct RerankingModelTests {
    @Test func rerankedSearchFindsTheRightDocumentAndReportsCrossEncoderScores() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory, residency: .keepResident)
            for item in corpus { try await kb.add(.text(item.text, title: item.title)) }
            let stats = await kb.stats
            #expect(stats.rerankerID?.hasPrefix("Qwen3 Reranker") == true)

            let results = try await kb.search("How many people left the study early?", options: RetrievalOptions(limit: 3, mode: .hybrid))
            #expect(results.first?.document.title == "trial")
            #expect(results.allSatisfy { $0.scores.rerank != nil && $0.scores.fused != nil })
            let scores = results.compactMap(\.scores.rerank)
            #expect(scores == scores.sorted(by: >), "results are ordered by the cross-encoder")
            #expect((scores.first ?? 0) > (scores.last ?? 0) + 1, "a relevant passage is clearly separated from irrelevant ones")

            let plain = try await kb.search("How many people left the study early?", options: RetrievalOptions(limit: 3, rerank: false))
            #expect(plain.allSatisfy { $0.scores.rerank == nil })
        }
    }

    @Test func lowMemoryResidencyGivesTheSameAnswersAsKeepingEverythingLoaded() async throws {
        try await withTemporaryDirectory { directory in
            let resident = try await openKnowledgeBase(directory.appendingPathComponent("a"), residency: .keepResident)
            let frugal = try await openKnowledgeBase(directory.appendingPathComponent("b"), residency: .lowMemory)
            for item in corpus {
                try await resident.add(.text(item.text, title: item.title))
                try await frugal.add(.text(item.text, title: item.title))  // embedder in, then out again for the reranker below
            }
            for question in ["When was the lighthouse built?", "What temperature should I bake at?"] {
                // Alternating forces the low-memory knowledge base to swap embedder ↔ reranker on every search.
                let a = try await resident.search(question, options: RetrievalOptions(limit: 3))
                let b = try await frugal.search(question, options: RetrievalOptions(limit: 3))
                #expect(a.map(\.document.title) == b.map(\.document.title), "\(question)")
                for (x, y) in zip(a.compactMap(\.scores.rerank), b.compactMap(\.scores.rerank)) { #expect(abs(x - y) < 0.05) }
            }
            await frugal.unloadModels()
            #expect(try await frugal.search("lighthouse tower", options: RetrievalOptions(limit: 1)).first?.document.title == "lighthouse")
        }
    }
}

@Suite("Model role errors through the public API", .enabled(if: modelsAvailable))
struct RoleErrorTests {
    @Test func aRerankerConfiguredAsTheEmbedderIsRejectedAsARoleMismatch() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await KnowledgeBase(
                directory: directory.appendingPathComponent("kb"), models: ModelSet(embedder: .file(ModelFixtures.fixtureReranker)),
                options: KnowledgeBaseOptions(gpuLayers: .none))
            do {
                _ = try await kb.add(.text("text", title: "t"))
                Issue.record("expected modelRoleMismatch")
            } catch LociError.modelRoleMismatch(_, let expected) {
                #expect(expected == "embedding")
            }
        }
    }

    @Test func anEmbeddingModelConfiguredAsTheRerankerIsRejectedWhenSearchNeedsIt() async throws {
        try await withTemporaryDirectory { directory in
            let models = ModelSet(embedder: .file(ModelFixtures.fixtureEmbedder), reranker: .file(ModelFixtures.fixtureEmbedder))
            let kb = try await KnowledgeBase(
                directory: directory.appendingPathComponent("kb"), models: models, options: KnowledgeBaseOptions(gpuLayers: .none))
            try await kb.add(.text("alpha beta gamma", title: "a"))
            try await kb.add(.text("alpha delta", title: "b"))
            do {
                _ = try await kb.search("alpha", options: RetrievalOptions(mode: .keyword))
                Issue.record("expected modelRoleMismatch")
            } catch LociError.modelRoleMismatch(_, let expected) {
                #expect(expected == "reranker")
            }
            #expect(try await kb.search("alpha", options: RetrievalOptions(mode: .keyword, rerank: false)).count == 2)
        }
    }
}
