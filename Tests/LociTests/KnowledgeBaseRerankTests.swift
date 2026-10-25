import Foundation
import Testing
@testable import Loci
@testable import LociLlama
@testable import LociStore

private func withKnowledgeBase(
    reranker: FakeReranker? = FakeReranker(), policy: ResidencyPolicy = .keepResident,
    memory: any MemoryProbe = SystemMemoryProbe(), estimatedBytes: UInt64 = 0,
    _ body: (KnowledgeBase, ModelServices) async throws -> Void
) async throws {
    let directory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let services = await ModelServices(
        embedder: FakeEmbedder(), reranker: reranker, policy: policy, memory: memory, estimatedBytes: estimatedBytes)
    let kb = try await KnowledgeBase(directory: directory.appendingPathComponent("kb"), services: services)
    try await body(kb, services)
}

private func seedNeedleCorpus(_ kb: KnowledgeBase) async throws {
    // Separate documents never merge, so each is its own passage.
    for (title, text) in [
        ("first", "Quarterly report about alpha and beta results across the regions."),
        ("second", "Alpha beta notes: the needle is mentioned in this passage about alpha beta."),
        ("third", "Alpha beta gamma delta epsilon, a long list of unrelated greek letters."),
    ] {
        try await kb.add(.text(text, title: title))
    }
}

@Suite("KnowledgeBase reranking")
struct KnowledgeBaseRerankTests {
    /// Prefers passages containing "needle", whatever the retrievers thought.
    private func needleReranker() -> FakeReranker {
        FakeReranker { _, passage in passage.lowercased().contains("needle") ? 10 : 0 }
    }

    @Test func rerankingChangesTheOrderAndReportsRerankScores() async throws {
        try await withKnowledgeBase(reranker: needleReranker()) { kb, _ in
            try await seedNeedleCorpus(kb)
            let base = try await kb.search("alpha beta", options: RetrievalOptions(limit: 3, mode: .hybrid, rerank: false))
            let reranked = try await kb.search("alpha beta", options: RetrievalOptions(limit: 3, mode: .hybrid, rerank: true))

            #expect(reranked.first?.document.title == "second")
            #expect(reranked.first?.scores.rerank == 10)
            #expect(reranked.dropFirst().allSatisfy { $0.scores.rerank == 0 })
            #expect(reranked.allSatisfy { $0.scores.fused != nil && $0.scores.bm25 != nil }, "earlier stage scores are kept")
            #expect(base.allSatisfy { $0.scores.rerank == nil })
            #expect(Set(base.map(\.document.title)) == Set(reranked.map(\.document.title)))
        }
    }

    @Test func everyModeCanBeReranked() async throws {
        try await withKnowledgeBase(reranker: needleReranker()) { kb, _ in
            try await seedNeedleCorpus(kb)
            for mode in [RetrievalMode.keyword, .vector, .hybrid] {
                let results = try await kb.search("alpha beta", options: RetrievalOptions(limit: 3, mode: mode))
                #expect(results.first?.document.title == "second", "\(mode)")
            }
        }
    }

    @Test func disablingRerankSkipsTheModelEntirely() async throws {
        let reranker = needleReranker()
        try await withKnowledgeBase(reranker: reranker) { kb, services in
            try await seedNeedleCorpus(kb)
            _ = try await kb.search("alpha beta", options: RetrievalOptions(rerank: false))
            #expect(reranker.callCount == 0)
            #expect(await services.manager.residentRoles == [.embedder], "the reranker was never loaded")
            _ = try await kb.search("alpha beta", options: RetrievalOptions(rerank: true))
            #expect(reranker.callCount == 1)
            #expect(await services.manager.residentRoles == [.embedder, .reranker])
        }
    }

    @Test func withoutARerankerTheRerankOptionIsIgnored() async throws {
        try await withKnowledgeBase(reranker: nil) { kb, _ in
            try await seedNeedleCorpus(kb)
            let results = try await kb.search("alpha beta", options: RetrievalOptions(rerank: true))
            #expect(!results.isEmpty && results.allSatisfy { $0.scores.rerank == nil })
            #expect(await kb.stats.rerankerID == nil)
        }
    }

    @Test func onlyTheTopCandidatesAreScored() async throws {
        let reranker = FakeReranker()
        try await withKnowledgeBase(reranker: reranker) { kb, _ in
            for i in 0..<40 { try await kb.add(.text("shared topic words in document number \(i)", title: "d\(i)")) }
            let results = try await kb.search("shared topic", options: RetrievalOptions(limit: 3, mode: .keyword))
            #expect(results.count == 3)
            #expect(reranker.scoredPassages.map(\.count) == [KnowledgeBase.rerankDepth])
        }
    }

    @Test func rerankerFailuresSurfaceAsLociErrors() async throws {
        struct Boom: Error {}
        let reranker = FakeReranker()
        reranker.failure = LlamaError.backend(operation: "decode", code: 5)
        try await withKnowledgeBase(reranker: reranker) { kb, _ in
            try await seedNeedleCorpus(kb)
            do {
                _ = try await kb.search("alpha beta", options: .default)
                Issue.record("expected an error")
            } catch LociError.backendFailure(_, let code) {
                #expect(code == 5)
            }
            #expect(try await kb.search("alpha beta", options: RetrievalOptions(rerank: false)).isEmpty == false, "search still works without reranking")
        }
    }

    @Test func statsReportTheConfiguredReranker() async throws {
        try await withKnowledgeBase(reranker: FakeReranker(id: "my-reranker|1024|42")) { kb, _ in
            #expect(await kb.stats.rerankerID == "my-reranker|1024|42")
        }
    }
}

@Suite("KnowledgeBase residency")
struct KnowledgeBaseResidencyTests {
    @Test func lowMemoryNeverHoldsTwoModelsAndStillAnswersCorrectly() async throws {
        let reranker = FakeReranker { _, passage in passage.lowercased().contains("needle") ? 10 : 0 }
        try await withKnowledgeBase(reranker: reranker, policy: .lowMemory) { kb, services in
            try await seedNeedleCorpus(kb)
            #expect(await services.manager.residentRoles == [.embedder], "ingest uses the embedder only")

            let results = try await kb.search("alpha beta", options: RetrievalOptions(limit: 3, mode: .hybrid))
            #expect(results.first?.document.title == "second")
            #expect(await services.manager.residentRoles == [.reranker], "embed the query, then swap in the reranker")

            try await kb.add(.text("another document about alpha", title: "fourth"))
            #expect(await services.manager.residentRoles == [.embedder], "ingest swaps the embedder back in")
        }
    }

    @Test func balancedKeepsBothSearchModelsResident() async throws {
        try await withKnowledgeBase(policy: .balanced) { kb, services in
            try await seedNeedleCorpus(kb)
            _ = try await kb.search("alpha beta", options: .default)
            #expect(await services.manager.residentRoles == [.embedder, .reranker])
        }
    }

    @Test func aSearchThatCannotFitThePhoneIsRefusedWithInsufficientMemoryAndKeywordStillWorks() async throws {
        let ledger = MemoryLedger(capacity: 1_000_000)
        try await withKnowledgeBase(memory: ledger, estimatedBytes: 600_000) { kb, services in
            try await seedNeedleCorpus(kb)  // embedder: 600 KB of 1 MB
            ledger.setCapacity(100_000)
            await kb.unloadModels()

            do {
                _ = try await kb.search("alpha beta", options: RetrievalOptions(mode: .hybrid))
                Issue.record("expected insufficientMemory")
            } catch LociError.insufficientMemory(let required, let available) {
                #expect(required == 600_000 && available == 100_000)
            }
            #expect(await services.manager.residentRoles.isEmpty, "nothing was loaded")
            // Keyword search needs no embedder, but it still reranks, so it is refused too unless rerank is off.
            await #expect(throws: LociError.self) { try await kb.search("alpha beta", options: RetrievalOptions(mode: .keyword)) }
            #expect(try await kb.search("alpha beta", options: RetrievalOptions(mode: .keyword, rerank: false)).isEmpty == false)

            ledger.setCapacity(1_000_000)
            #expect(try await kb.search("alpha beta", options: RetrievalOptions(mode: .hybrid)).isEmpty == false)
        }
    }

    @Test func unloadModelsReleasesEveryModel() async throws {
        try await withKnowledgeBase(policy: .keepResident) { kb, services in
            try await seedNeedleCorpus(kb)
            _ = try await kb.search("alpha", options: .default)
            #expect(await services.manager.residentRoles == [.embedder, .reranker])
            await kb.unloadModels()
            #expect(await services.manager.residentRoles.isEmpty)
        }
    }

    @Test func theResidencyOptionMapsToThePolicy() {
        #expect(ModelResidency.lowMemory.policy == .lowMemory)
        #expect(ModelResidency.balanced.policy == .balanced)
        #expect(ModelResidency.keepResident.policy == .keepResident)
    }
}
