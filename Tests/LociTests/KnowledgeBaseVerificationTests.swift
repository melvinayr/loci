import Foundation
import Testing
@testable import Loci
@testable import LociGrounding
@testable import LociLlama

/// A cross-encoder stand-in: the logit rises by 2 for every distinctive word the claim shares with the passage, starting at −3,
/// so 0 shared words ≈ 5 %, 1 ≈ 27 %, 2 ≈ 73 %, 3 ≈ 95 %.
private func overlapReranker(_ id: String = "overlap-reranker|1|0") -> FakeReranker {
    FakeReranker(id: id) { claim, passage in
        func words(_ text: String) -> Set<String> {
            Set(text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init).filter { $0.count > 3 })
        }
        return Float(words(claim).intersection(words(passage)).count) * 2 - 3
    }
}

private func withKnowledgeBase(
    reply: String, reranker: FakeReranker? = overlapReranker(), policy: ResidencyPolicy = .keepResident,
    _ body: (KnowledgeBase, FakeGenerator, ModelServices) async throws -> Void
) async throws {
    let directory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let generator = FakeGenerator(reply)
    let services = await ModelServices(embedder: FakeEmbedder(), reranker: reranker, generator: generator, policy: policy)
    let kb = try await KnowledgeBase(directory: directory.appendingPathComponent("kb"), services: services)
    try await kb.add(.text("The trial enrolled 412 participants across nine clinical sites.", title: "trial"))
    try await kb.add(.text("Nausea was the most common adverse event reported by patients.", title: "safety"))
    try await kb.add(.text("The lighthouse tower is 38 metres tall and made of granite.", title: "lighthouse"))
    try await body(kb, generator, services)
}

private let options = AnswerOptions(retrieval: RetrievalOptions(limit: 3, mode: .keyword, rerank: false))

@Suite("Answer verification in the knowledge base")
struct KnowledgeBaseVerificationTests {
    @Test func aFabricatedSentenceInAGoodAnswerIsFlaggedWhileTheRestIsSupported() async throws {
        // The second sentence cites a real source but claims something the source never says.
        try await withKnowledgeBase(reply: "The trial enrolled 412 participants across nine clinical sites [1]. The study cost three million dollars [1].") { kb, _, _ in
            let answer = try await kb.ask("trial participants clinical sites", options: options)

            #expect(!answer.refused)
            let labels = answer.sentences.map(\.support)
            #expect(labels == [.supported, .unsupported], "got \(answer.sentences.map { ($0.support, $0.score) })")
            #expect(answer.sentences[0].score > 0.9 && answer.sentences[1].score < 0.35)
            #expect(answer.sentences.map(\.citationIndices) == [[1], [1]])
            let summary = try #require(answer.support)
            #expect(summary.supportedFraction == 0.5 && summary.level == .unsupported, "worst case rolls up")
            #expect(answer.text.contains("The study cost three million dollars [1]."), "the flagged sentence is kept, not hidden")
        }
    }

    @Test func aFullySupportedAnswerRollsUpAsSupported() async throws {
        try await withKnowledgeBase(reply: "The trial enrolled 412 participants across nine clinical sites [1].") { kb, _, _ in
            let answer = try await kb.ask("trial participants", options: options)
            let summary = try #require(answer.support)
            #expect(summary.supportedFraction == 1 && summary.level == .supported)
            #expect(answer.stats.verificationTime > .zero)
        }
    }

    @Test func sentenceEventsComeAfterTheTokensAndBeforeFinished() async throws {
        try await withKnowledgeBase(reply: "The trial enrolled 412 participants across nine clinical sites [1]. Nausea was the most common adverse event [2].") { kb, _, _ in
            var order: [String] = []
            var sentences: [AnsweredSentence] = []
            var finished: Answer?
            for try await event in kb.streamAnswer("trial nausea", options: options) {
                switch event {
                case .retrieved: order.append("retrieved")
                case .token: if order.last != "token" { order.append("token") }
                case .sentence(let sentence): order.append("sentence"); sentences.append(sentence)
                case .finished(let answer): order.append("finished"); finished = answer
                }
            }
            #expect(order == ["retrieved", "token", "sentence", "sentence", "finished"])
            #expect(sentences.count == 2 && sentences.map(\.text) == finished?.sentences.map(\.text))
            #expect(sentences.map(\.citationIndices) == finished?.sentences.map(\.citationIndices))
        }
    }

    @Test func aMissingCitationIsRecoveredFromTheRetrievedPassages() async throws {
        try await withKnowledgeBase(reply: "The trial enrolled 412 participants across nine clinical sites.") { kb, _, _ in
            let answer = try await kb.ask("trial participants", options: options)
            #expect(answer.text == "The trial enrolled 412 participants across nine clinical sites [1].")
            #expect(answer.sentences.first?.support == .supported && answer.sentences.first?.citationIndices == [1])
            let citation = try #require(answer.citations.first)
            #expect(citation.index == 1 && citation.document.title == "trial")
            let text = try #require(try await kb.documentText(for: citation.document.id))
            #expect(text.scalarSlice(citation.range) == citation.excerpt)
        }
    }

    @Test func anUncitedClaimNoSourceSupportsStaysUncitedAndUnsupported() async throws {
        try await withKnowledgeBase(reply: "The trial enrolled 412 participants across nine clinical sites [1]. Penguins migrate every winter.") { kb, _, _ in
            let answer = try await kb.ask("trial participants", options: options)
            #expect(answer.sentences.map(\.support) == [.supported, .unsupported])
            #expect(answer.sentences[1].citationIndices.isEmpty)
            #expect(answer.text.hasSuffix("Penguins migrate every winter."))
        }
    }

    @Test func aMajorityOfUnsupportedSentencesTurnsTheAnswerIntoARefusalThatListsTheSources() async throws {
        try await withKnowledgeBase(reply: "Penguins migrate every winter [1]. Volcanoes erupt without warning [1]. The trial enrolled 412 participants across nine clinical sites [1].") { kb, _, _ in
            var sentenceEvents = 0
            var finished: Answer?
            for try await event in kb.streamAnswer("trial participants", options: options) {
                if case .sentence = event { sentenceEvents += 1 }
                if case .finished(let answer) = event { finished = answer }
            }
            let answer = try #require(finished)
            #expect(sentenceEvents == 3, "sentences were reported as they were checked")
            #expect(answer.refused && answer.text == GroundingPrompt.refusal)
            #expect(answer.sentences.isEmpty && answer.support == nil)
            #expect(answer.citations.map(\.index) == Array(1...answer.citations.count), "the retrieved sources, for inspection")
            #expect(answer.stats.verificationTime > .zero)
        }
    }

    @Test func refuseWhenUnsupportedCanBeSwitchedOffToSeeTheFlaggedAnswer() async throws {
        try await withKnowledgeBase(reply: "Penguins migrate every winter [1]. Volcanoes erupt without warning [1].") { kb, _, _ in
            var lenient = options
            lenient.refuseWhenUnsupported = false
            let answer = try await kb.ask("trial participants", options: lenient)
            #expect(!answer.refused)
            #expect(answer.sentences.map(\.support) == [.unsupported, .unsupported])
            #expect(answer.support?.level == .unsupported && answer.support?.supportedFraction == 0)
        }
    }

    @Test func aFabricatedCitationNumberMakesTheSentenceUnsupported() async throws {
        try await withKnowledgeBase(reply: "The trial enrolled 412 participants across nine clinical sites [1]. The tower is 38 metres tall [9].") { kb, _, _ in
            let answer = try await kb.ask("trial participants tower", options: options)
            #expect(answer.sentences.map(\.support) == [.supported, .unsupported])
            #expect(!answer.text.contains("[9]"))
            #expect(answer.sentences[1].citationIndices.isEmpty)
        }
    }

    @Test func thresholdsDecideTheBands() async throws {
        // Three shared words → logit 3 → a probability of about 0.95.
        let reply = "The trial enrolled 412 participants [1]."
        try await withKnowledgeBase(reply: reply) { kb, _, _ in
            var strict = options
            strict.supportThresholds = (supported: 0.999, weak: 0.5)
            let weak = try await kb.ask("trial participants", options: strict)
            #expect(weak.sentences.first?.support == .weak)
            strict.supportThresholds = (supported: 0.999, weak: 0.9999)
            strict.refuseWhenUnsupported = false
            let unsupported = try await kb.ask("trial participants", options: strict)
            #expect(unsupported.sentences.first?.support == .unsupported)
        }
    }

    @Test func aRefusalFromTheModelIsNotVerified() async throws {
        let reranker = overlapReranker()
        try await withKnowledgeBase(reply: "I couldn't find the answer in your documents.", reranker: reranker) { kb, _, _ in
            let answer = try await kb.ask("trial participants", options: options)
            #expect(answer.refused && reranker.callCount == 0)
            #expect(answer.support == nil)
        }
    }

    @Test func turningVerificationOffSkipsScoringAndSentenceEvents() async throws {
        let reranker = overlapReranker()
        try await withKnowledgeBase(reply: "The trial enrolled 412 participants across nine clinical sites [1].", reranker: reranker) { kb, _, _ in
            var off = options
            off.verify = false
            var events = 0
            for try await event in kb.streamAnswer("trial participants", options: off) { if case .sentence = event { events += 1 } }
            #expect(events == 0 && reranker.callCount == 0)
            let answer = try await kb.ask("trial participants", options: off)
            #expect(answer.support == nil && answer.stats.verificationTime == .zero)
        }
    }

    @Test func withoutARerankerEmbeddingSimilarityIsTheFallbackScorer() async throws {
        try await withKnowledgeBase(
            reply: "The trial enrolled 412 participants across nine clinical sites [1]. Quantum entanglement powers submarines [1].", reranker: nil
        ) { kb, _, _ in
            var cosine = options
            cosine.supportThresholds = (supported: 0.5, weak: 0.25)  // cosine scores live on a different scale
            cosine.refuseWhenUnsupported = false
            let answer = try await kb.ask("trial participants", options: cosine)
            #expect(answer.sentences.map(\.support) == [.supported, .unsupported], "got \(answer.sentences.map(\.score))")
            #expect(answer.sentences.allSatisfy { (0...1).contains($0.score) })
        }
    }

    @Test func supportScoresArePublicAndOnAZeroToOneScale() async throws {
        try await withKnowledgeBase(reply: "x") { kb, _, _ in
            let scores = try await kb.supportScores(
                for: "The trial enrolled 412 participants",
                against: ["The trial enrolled 412 participants across nine clinical sites.", "The lighthouse tower is made of granite."])
            #expect(scores.count == 2 && scores[0] > 0.9 && scores[1] < 0.1)
            #expect(try await kb.supportScores(for: "x", against: []).isEmpty)
        }
        try await withKnowledgeBase(reply: "x", reranker: nil) { kb, _, _ in
            let scores = try await kb.supportScores(for: "trial participants enrolled", against: ["The trial enrolled participants.", "granite lighthouse tower"])
            #expect(scores.allSatisfy { (0...1).contains($0) } && scores[0] > scores[1])
        }
    }

    @Test func probabilityMapsLogitsToZeroOne() {
        #expect(KnowledgeBase.probability(0) == 0.5)
        #expect(KnowledgeBase.probability(40) > 0.999 && KnowledgeBase.probability(-40) < 0.001)
        #expect(KnowledgeBase.probability(2) > KnowledgeBase.probability(1))
        #expect(KnowledgeBase.probability(.infinity).isFinite && KnowledgeBase.probability(-.infinity).isFinite)
    }

    @Test func verificationSwapsTheRerankerInUnderLowMemory() async throws {
        try await withKnowledgeBase(reply: "The trial enrolled 412 participants across nine clinical sites [1].", policy: .lowMemory) { kb, _, services in
            _ = try await kb.ask("trial participants", options: AnswerOptions(retrieval: RetrievalOptions(limit: 3, mode: .keyword, rerank: false)))
            #expect(await services.manager.residentRoles == [.reranker], "generator out, reranker in for verification")
        }
    }

    @Test func aFailingScorerSurfacesAsALociError() async throws {
        let reranker = overlapReranker()
        reranker.failure = LlamaError.backend(operation: "decode", code: 4)
        try await withKnowledgeBase(reply: "The trial enrolled 412 participants across nine clinical sites [1].", reranker: reranker) { kb, _, _ in
            do {
                _ = try await kb.ask("trial participants", options: options)
                Issue.record("expected an error")
            } catch LociError.backendFailure(_, let code) {
                #expect(code == 4)
            }
        }
    }
}

@Suite("Document text and page accessors")
struct DocumentAccessorTests {
    @Test func documentTextIsWhatRangesReferTo() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let kb = try await KnowledgeBase(directory: directory.appendingPathComponent("kb"), services: await ModelServices(embedder: FakeEmbedder()))
        let note = try await kb.add(.text("Alpha beta.\n\nGamma delta.", title: "note"))
        #expect(try await kb.documentText(for: note.id) == "Alpha beta.\n\nGamma delta.")
        #expect(try await kb.pageRanges(for: note.id) == nil, "only PDFs have pages")
        #expect(try await kb.documentText(for: DocumentID(rawValue: "missing")) == nil)

        let pdf = directory.appendingPathComponent("two.pdf")
        try writePDF(pages: ["First page words.", "Second page words."], to: pdf)
        let document = try await kb.add(.file(pdf))
        let ranges = try #require(try await kb.pageRanges(for: document.id))
        let text = try #require(try await kb.documentText(for: document.id))
        #expect(ranges.count == 2)
        let first = try #require(ranges[0]), second = try #require(ranges[1])
        #expect(text.scalarSlice(first).contains("First"))
        #expect(text.scalarSlice(second).contains("Second"))
    }
}
