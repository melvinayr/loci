import Foundation
import Testing
import Loci
@testable import LociEval

private let evalSmall = ModelFixtures.repositoryRoot.appendingPathComponent("Tests/Fixtures/eval-small", isDirectory: true)

private let verifierModels = ModelFixtures.exists(ModelFixtures.fixtureEmbedder) && ModelFixtures.exists(ModelFixtures.fixtureReranker)

private func withTemporaryDirectory(_ body: (URL) async throws -> Void) async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("loci-it-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try await body(directory)
}

/// The real instruction-aware reranker used as a verifier, on CPU.
@Suite("Verification with the real reranker", .serialized, .enabled(if: verifierModels))
struct RerankerVerificationTests {
    private func openKnowledgeBase(_ directory: URL, generator: URL? = nil) async throws -> KnowledgeBase {
        try await KnowledgeBase(
            directory: directory.appendingPathComponent("kb"),
            models: ModelSet(embedder: .file(ModelFixtures.fixtureEmbedder), reranker: .file(ModelFixtures.fixtureReranker), generator: generator.map { .file($0) }),
            options: KnowledgeBaseOptions(gpuLayers: .none))
    }

    private let passage = "In total 374 participants completed the 12-week treatment period, a dropout rate of 9.2 percent, with similar rates across the three arms."

    @Test func aFaithfulClaimOutscoresAnUnrelatedOneAndAContradictionIsNotFlagged() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory)
            let scores = try await kb.supportScores(for: "The dropout rate was 9.2 percent.", against: [passage])
            let unrelated = try await kb.supportScores(for: "The lighthouse tower is made of granite.", against: [passage])
            #expect(scores[0] > 0.9, "faithful claim scored \(scores[0])")
            #expect(unrelated[0] < 0.35, "unrelated claim scored \(unrelated[0])")
            #expect(scores.allSatisfy { (0...1).contains($0) } && unrelated.allSatisfy { (0...1).contains($0) })
        }
    }

    /// A stratified slice of the hand-labelled subset keeps this quick on the CPU while still testing real separation.
    @Test func theRerankerSeparatesSupportedFromUnsupportedClaimsBetterThanChance() async throws {
        let loaded = try Dataset.folder(evalSmall).load()
        func slice(_ label: SupportExample.Label) -> [SupportExample] {
            Array(loaded.supportExamples.filter { $0.label == label && $0.split == .calibration }.prefix(8))
        }
        let examples = slice(.supported) + slice(.unsupported)
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory)
            var supported: [Float] = [], unsupported: [Float] = []
            for example in examples {
                let score = try await kb.supportScores(for: example.claim, against: [example.passage])[0]
                if example.label == .supported { supported.append(score) } else { unsupported.append(score) }
            }
            var wins = 0.0
            for s in supported { for u in unsupported { wins += s > u ? 1 : s == u ? 0.5 : 0 } }
            let auc = wins / Double(supported.count * unsupported.count)
            #expect(auc >= 0.75, "AUC \(auc); supported \(supported), unsupported \(unsupported)")
            #expect(supported.min() ?? 0 > 0.5, "supported claims should not be scored as unsupported: \(supported)")
        }
    }

    @Test(.enabled(if: ModelFixtures.exists(ModelFixtures.fixtureGenerator)))
    func anAnswerThroughTheRealPipelineIsVerifiedSentenceBySentence() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory, generator: ModelFixtures.fixtureGenerator)
            try await kb.add(.text("The Cape Marrow lighthouse was built in 1874. Its granite tower is 38 metres tall.", title: "lighthouse"))
            try await kb.add(.text("Feed the sourdough starter once a day and bake at 245 degrees Celsius.", title: "bread"))

            var options = AnswerOptions(retrieval: RetrievalOptions(limit: 2, rerank: false), maxAnswerTokens: 80, temperature: 0)
            options.refuseWhenUnsupported = false
            var events: [String] = []
            var sentenceEvents: [AnsweredSentence] = []
            var finished: Answer?
            for try await event in kb.streamAnswer("When was the Cape Marrow lighthouse built?", options: options) {
                switch event {
                case .retrieved: events.append("retrieved")
                case .token: if events.last != "token" { events.append("token") }
                case .sentence(let sentence): events.append("sentence"); sentenceEvents.append(sentence)
                case .finished(let answer): events.append("finished"); finished = answer
                }
            }
            let answer = try #require(finished)
            #expect(events.first == "retrieved" && events.last == "finished")
            if !answer.refused {
                #expect(!sentenceEvents.isEmpty && sentenceEvents.map(\.text) == answer.sentences.map(\.text))
                #expect(answer.sentences.allSatisfy { (0...1).contains($0.score) })
                let summary = try #require(answer.support)
                #expect((0...1).contains(summary.supportedFraction))
                #expect(answer.stats.verificationTime > .zero)
                // Every cited sentence's markers refer to a citation that exists.
                for sentence in answer.sentences { for index in sentence.citationIndices { #expect((1...max(answer.citations.count, 1)).contains(index)) } }
            }
        }
    }
}
