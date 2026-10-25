import Foundation
import Testing
import Loci

private func withTemporaryDirectory(_ body: (URL) async throws -> Void) async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("loci-it-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try await body(directory)
}

private let corpus: [(title: String, text: String)] = [
    ("trial", "The Brindlemoor trial enrolled 412 participants at nine sites. In total 374 participants completed the treatment period, a dropout rate of 9.2 percent."),
    ("lighthouse", "The Cape Marrow lighthouse was built in 1874. Its granite tower is 38 metres tall and the light was converted to electricity in 1932."),
    ("bread", "Feed the sourdough starter at a ratio of 1:1:1 by weight and bake the loaf at 245 degrees Celsius in a Dutch oven."),
]

private func openKnowledgeBase(
    _ directory: URL, generator: URL, reranker: URL? = nil, residency: ModelResidency = .keepResident
) async throws -> KnowledgeBase {
    let kb = try await KnowledgeBase(
        directory: directory.appendingPathComponent("kb"),
        models: ModelSet(embedder: .file(ModelFixtures.fixtureEmbedder), reranker: reranker.map { .file($0) }, generator: .file(generator)),
        options: KnowledgeBaseOptions(residency: residency, gpuLayers: .none))
    for item in corpus { try await kb.add(.text(item.text, title: item.title)) }
    return kb
}

private let deterministic = AnswerOptions(
    retrieval: RetrievalOptions(limit: 3, rerank: false), maxAnswerTokens: 120, temperature: 0)

private let fixturesPresent = ModelFixtures.exists(ModelFixtures.fixtureEmbedder) && ModelFixtures.exists(ModelFixtures.fixtureGenerator)

/// Retrieval-augmented answers with a real embedder and a real (small) chat model, on CPU.
@Suite("Answers with Qwen3-0.6B", .serialized, .enabled(if: fixturesPresent))
struct QwenAnswerTests {
    @Test func answersFromTheRightDocumentWithValidCitations() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory, generator: ModelFixtures.fixtureGenerator)
            let answer = try await kb.ask("When was the Cape Marrow lighthouse built?", options: deterministic)

            #expect(!answer.refused)
            #expect(answer.text.contains("1874"), "got: \(answer.text)")
            #expect(!answer.text.contains("<think>") && !answer.text.contains("<|im_"))
            #expect(answer.support != nil && !answer.sentences.isEmpty, "answers are verified by default")
            #expect(answer.citations.first?.document.title == "lighthouse")
            #expect(answer.citations.map(\.index) == Array(1...max(answer.citations.count, 1)).prefix(answer.citations.count).map { $0 })
            for citation in answer.citations {
                #expect(citation.excerpt.contains("1874") || citation.excerpt.contains("lighthouse"))
                #expect(citation.range.count == citation.excerpt.unicodeScalars.count)
            }
            #expect(answer.stats.contextTokens > 100 && answer.stats.generationTokensPerSecond > 0)
        }
    }

    @Test func aQuestionTheDocumentsCannotAnswerIsRefused() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory, generator: ModelFixtures.fixtureGenerator)
            let answer = try await kb.ask("Who is the mayor of Aldermere?", options: deterministic)
            #expect(answer.refused, "got: \(answer.text)")
            #expect(answer.support == nil && answer.sentences.isEmpty)
            #expect(answer.citations.map(\.index) == Array(1...max(answer.citations.count, 1)).prefix(answer.citations.count).map { $0 },
                    "a refusal lists the retrieved passages for inspection")
        }
    }

    @Test func streamingDeliversSourcesFirstThenTokensThenTheAnswer() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory, generator: ModelFixtures.fixtureGenerator)
            var events: [String] = []
            var streamed = ""
            var final: Answer?
            for try await event in kb.streamAnswer("How tall is the tower?", options: deterministic) {
                switch event {
                case .retrieved(let results): events.append("retrieved:\(results.count)")
                case .token(let piece): if events.last != "token" { events.append("token") }; streamed += piece
                case .sentence: events.append("sentence")
                case .finished(let answer): events.append("finished"); final = answer
                }
            }
            #expect(events.first?.hasPrefix("retrieved:") == true && events.dropFirst().first == "token" && events.last == "finished")
            #expect(!streamed.contains("\u{FFFD}") && !streamed.contains("<think>"))
            let answer = try #require(final)
            #expect(answer.text.contains("38") || streamed.contains("38"))
        }
    }

    @Test func lowMemoryResidencyAnswersLikeKeepingEverythingResident() async throws {
        guard ModelFixtures.exists(ModelFixtures.fixtureReranker) else { return }
        try await withTemporaryDirectory { directory in
            var options = deterministic
            options.retrieval.rerank = true
            let resident = try await openKnowledgeBase(
                directory.appendingPathComponent("a"), generator: ModelFixtures.fixtureGenerator, reranker: ModelFixtures.fixtureReranker, residency: .keepResident)
            let frugal = try await openKnowledgeBase(
                directory.appendingPathComponent("b"), generator: ModelFixtures.fixtureGenerator, reranker: ModelFixtures.fixtureReranker, residency: .lowMemory)
            let question = "What temperature should I bake the loaf at?"
            let a = try await resident.ask(question, options: options)
            let b = try await frugal.ask(question, options: options)
            // Swapping models in and out must not change what is answered (greedy decoding on CPU is deterministic).
            // Whether a 0.6B model manages to answer this particular question is not what is being tested.
            #expect(a.text == b.text, "\(a.text) ≠ \(b.text)")
            #expect(a.refused == b.refused && a.citations.map(\.chunkID) == b.citations.map(\.chunkID))
            #expect(b.stats.rerankTime > .zero)
        }
    }

    @Test func aCancelledAnswerLeavesTheKnowledgeBaseUsable() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory, generator: ModelFixtures.fixtureGenerator)
            var long = deterministic
            long.maxAnswerTokens = 1_000
            let task = Task { for try await _ in kb.streamAnswer("Describe everything in the documents in great detail.", options: long) {} }
            try await Task.sleep(for: .milliseconds(1_500))
            task.cancel()
            _ = await task.result
            let again = try await kb.ask("When was the Cape Marrow lighthouse built?", options: deterministic)
            #expect(again.text.contains("1874"))
        }
    }

    @Test func noGeneratorMeansNoAnswers() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await KnowledgeBase(
                directory: directory.appendingPathComponent("kb"), models: ModelSet(embedder: .file(ModelFixtures.fixtureEmbedder)),
                options: KnowledgeBaseOptions(gpuLayers: .none))
            try await kb.add(.text("something", title: "t"))
            do {
                _ = try await kb.ask("something?")
                Issue.record("expected noGenerator")
            } catch LociError.noGenerator {
            }
        }
    }

    @Test func aRerankerOrEmbedderConfiguredAsTheGeneratorIsRejected() async throws {
        try await withTemporaryDirectory { directory in
            for bad in [ModelFixtures.fixtureReranker, ModelFixtures.fixtureEmbedder].filter(ModelFixtures.exists) {
                let kb = try await KnowledgeBase(
                    directory: directory.appendingPathComponent(UUID().uuidString),
                    models: ModelSet(embedder: .file(ModelFixtures.fixtureEmbedder), generator: .file(bad)),
                    options: KnowledgeBaseOptions(gpuLayers: .none))
                try await kb.add(.text("The tower is 38 metres tall.", title: "t"))
                do {
                    _ = try await kb.ask("How tall?", options: deterministic)
                    Issue.record("expected modelRoleMismatch for \(bad.lastPathComponent)")
                } catch LociError.modelRoleMismatch(_, let expected) {
                    #expect(expected == "generator")
                }
            }
        }
    }
}

/// Gemma 4 needs the fallback chat template and carries a BF16 tensor; this runs the whole pipeline with it.
@Suite("Answers with Gemma 4", .serialized, .enabled(if: fixturesPresent && ModelFixtures.exists(ModelFixtures.generator)))
struct GemmaAnswerTests {
    @Test func answersWithCitationsAndRefusesWhatItCannotAnswer() async throws {
        try await withTemporaryDirectory { directory in
            let kb = try await openKnowledgeBase(directory, generator: ModelFixtures.generator)
            var options = deterministic
            options.retrieval.limit = 2
            options.maxAnswerTokens = 60

            let answer = try await kb.ask("How many participants were enrolled in the trial?", options: options)
            #expect(answer.text.contains("412"), "got: \(answer.text)")
            #expect(answer.citations.first?.document.title == "trial")
            #expect(!answer.text.contains("<turn|>") && !answer.text.contains("<|channel>"))

            let refusal = try await kb.ask("Who is the mayor of Aldermere?", options: options)
            #expect(refusal.refused, "got: \(refusal.text)")
        }
    }
}
