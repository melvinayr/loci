import Foundation
import os
import Testing
@testable import LociGrounding

/// Scores by looking up the claim: any registered keyword found in the claim decides the per-passage scores.
private final class ScriptedScorer: Sendable {
    private let rules: [(keyword: String, scores: [Float])]
    private let calls = OSAllocatedUnfairLock(initialState: [(claim: String, passages: [String])]())

    init(_ rules: [(String, [Float])]) { self.rules = rules.map { (keyword: $0.0, scores: $0.1) } }

    var recorded: [(claim: String, passages: [String])] { calls.withLock { $0 } }

    var scorer: SupportScorer {
        { [self] claim, passages in
            calls.withLock { $0.append((claim, passages)) }
            guard let rule = rules.first(where: { claim.lowercased().contains($0.keyword.lowercased()) }) else { return passages.map { _ in 0 } }
            return (0..<passages.count).map { $0 < rule.scores.count ? rule.scores[$0] : 0 }
        }
    }
}

private let defaults = SupportThresholds(supported: 0.6, weak: 0.35)

@Suite("Support thresholds")
struct SupportThresholdTests {
    @Test func scoresFallIntoBands() {
        let t = SupportThresholds(supported: 0.6, weak: 0.35)
        #expect(t.label(for: 0.9) == .supported && t.label(for: 0.6) == .supported)
        #expect(t.label(for: 0.59) == .weak && t.label(for: 0.35) == .weak)
        #expect(t.label(for: 0.34) == .unsupported && t.label(for: 0) == .unsupported)
    }

    @Test func misorderedThresholdsAreKeptOrdered() {
        let t = SupportThresholds(supported: 0.2, weak: 0.5)
        #expect(t.supported == 0.5 && t.weak == 0.5)
        #expect(t.label(for: 0.5) == .supported && t.label(for: 0.49) == .unsupported)
    }
}

@Suite("Sentence splitting")
struct SentenceSplitterTests {
    private func sentences(_ text: String) -> [String] {
        SentenceSplitter.split(text).map { String(text[$0.range]) }
    }

    @Test func splitsSentencesAndKeepsTheirMarkers() {
        #expect(sentences("The trial enrolled 412 people [1]. Nausea was common [2][3].") ==
                ["The trial enrolled 412 people [1].", "Nausea was common [2][3]."])
    }

    @Test func aMarkerAfterTheFullStopStaysWithItsSentence() {
        #expect(sentences("The trial enrolled 412 people. [1] Nausea was common. [2]") ==
                ["The trial enrolled 412 people. [1]", "Nausea was common. [2]"])
    }

    @Test func listItemsAreSeparateSentencesAndTheirNumbersStayWithThem() {
        let text = "1. First fact [1].\n2. Second fact [2].\n- Third fact [1]."
        #expect(sentences(text) == ["1. First fact [1].", "2. Second fact [2].", "- Third fact [1]."])
        for (index, range) in SentenceSplitter.split(text).enumerated() {
            let body = SentenceSplitter.body(of: range.range, in: text)
            #expect(body == ["First fact.", "Second fact.", "Third fact."][index])
        }
    }

    @Test func bodyRemovesMarkersAndTidiesSpacing() {
        let text = "Cats purr [1] and dogs bark [2][3] ."
        let range = SentenceSplitter.split(text)[0].range
        #expect(SentenceSplitter.body(of: range, in: text) == "Cats purr and dogs bark.")
    }

    @Test func segmentsWithNoWordsAreDropped() {
        #expect(sentences("[1]\n...\nReal sentence.") == ["Real sentence."])
        #expect(sentences("   \n\n  ").isEmpty)
        #expect(sentences("").isEmpty)
    }

    @Test func rangesPointIntoTheOriginalTextSoItCanBeRebuilt() {
        let text = "First [1].\n\nSecond   one [2]. Third!"
        var rebuilt = ""
        var cursor = text.startIndex
        for sentence in SentenceSplitter.split(text) {
            rebuilt += text[cursor..<sentence.range.lowerBound] + text[sentence.range]
            cursor = sentence.range.upperBound
        }
        rebuilt += text[cursor...]
        #expect(rebuilt == text)
    }
}

@Suite("Answer verification")
struct AnswerVerifierTests {
    private let sources = ["Source one text.", "Source two text.", "Source three text."]

    @Test func supportedWeakAndUnsupportedSentencesAreClassifiedByTheirBestCitedSource() async throws {
        let scorer = ScriptedScorer([("alpha", [0.9]), ("beta", [0.5]), ("gamma", [0.1])])
        let result = try await AnswerVerifier.verify(
            "Alpha is true [1]. Beta is partly true [1]. Gamma is made up [1].", sources: sources, thresholds: defaults, scorer: scorer.scorer)
        #expect(result.sentences.map(\.label) == [.supported, .weak, .unsupported])
        #expect(result.sentences.map(\.score) == [0.9, 0.5, 0.1])
        #expect(abs(result.supportedFraction - 1.0 / 3) < 1e-9)
        #expect(result.overall == .unsupported)
        #expect(!result.mostlyUnsupported, "one of three is not a majority")
    }

    @Test func aSentenceWithSeveralCitationsTakesTheBestScore() async throws {
        let scorer = ScriptedScorer([("alpha", [0.2, 0.8, 0.4])])
        let result = try await AnswerVerifier.verify("Alpha is true [1][2][3].", sources: sources, thresholds: defaults, scorer: scorer.scorer)
        #expect(result.sentences[0].score == 0.8 && result.sentences[0].label == .supported)
        #expect(scorer.recorded.first?.passages == sources, "all three cited passages were scored together")
    }

    @Test func citationsAreRenumberedByFirstUseAcrossSentences() async throws {
        let scorer = ScriptedScorer([("alpha", [0, 0, 0.9]), ("beta", [0.9, 0, 0])])
        let result = try await AnswerVerifier.verify("Alpha holds [3]. Beta holds [1]. Alpha again [3].", sources: sources, thresholds: defaults, scorer: scorer.scorer)
        #expect(result.text == "Alpha holds [1]. Beta holds [2]. Alpha again [1].")
        #expect(result.usedSources == [3, 1])
        #expect(result.sentences.map(\.citationIndices) == [[1], [2], [1]])
        #expect(result.sentences.map(\.citedSources) == [[3], [1], [3]])
    }

    @Test func aMissingCitationIsRecoveredWhenASourceSupportsTheSentence() async throws {
        let scorer = ScriptedScorer([("alpha", [0.1, 0.85, 0.2])])
        let result = try await AnswerVerifier.verify("Alpha is true.", sources: sources, thresholds: defaults, scorer: scorer.scorer)
        let sentence = result.sentences[0]
        #expect(sentence.recovered && sentence.label == .supported && sentence.citedSources == [2] && sentence.citationIndices == [1])
        #expect(sentence.text == "Alpha is true [1].", "the marker goes before the full stop")
        #expect(result.text == "Alpha is true [1]." && result.usedSources == [2])
        #expect(scorer.recorded.first?.passages == sources, "an uncited sentence is scored against every source")
    }

    @Test func recoveryDoesNotDisturbTheOrderOfFirstUse() async throws {
        let scorer = ScriptedScorer([("alpha", [0.9, 0, 0]), ("beta", [0, 0, 0.9]), ("gamma", [0, 0.95, 0])])
        let result = try await AnswerVerifier.verify("Alpha [1]. Beta is uncited. Gamma [2].", sources: sources, thresholds: defaults, scorer: scorer.scorer)
        #expect(result.text == "Alpha [1]. Beta is uncited [2]. Gamma [3].")
        #expect(result.usedSources == [1, 3, 2])
        #expect(result.sentences.map(\.recovered) == [false, true, false])
    }

    @Test func anUncitedSentenceNoSourceSupportsIsUnsupportedWithItsBestScoreReported() async throws {
        let scorer = ScriptedScorer([("alpha", [0.2, 0.5, 0.3])])  // best 0.5: weak, but it carries no citation
        let result = try await AnswerVerifier.verify("Alpha is claimed.", sources: sources, thresholds: defaults, scorer: scorer.scorer)
        let sentence = result.sentences[0]
        #expect(sentence.label == .unsupported && !sentence.recovered && sentence.citationIndices.isEmpty)
        #expect(sentence.score == 0.5)
        #expect(result.text == "Alpha is claimed.")
    }

    @Test func aSentenceWhoseMarkersAreAllInvalidIsUnsupportedAndNotScored() async throws {
        let scorer = ScriptedScorer([("alpha", [0.9, 0.9, 0.9])])
        let result = try await AnswerVerifier.verify("Alpha is true [7].", sources: sources, thresholds: defaults, scorer: scorer.scorer)
        #expect(result.sentences[0].label == .unsupported && !result.sentences[0].recovered && result.sentences[0].score == 0)
        #expect(result.text == "Alpha is true.")
        #expect(scorer.recorded.isEmpty, "a fabricated citation is not given a chance to be rescued")
    }

    @Test func invalidMarkersAlongsideValidOnesAreDroppedAndTheRestScored() async throws {
        let scorer = ScriptedScorer([("alpha", [0.9])])
        let result = try await AnswerVerifier.verify("Alpha is true [1] [9] .", sources: sources, thresholds: defaults, scorer: scorer.scorer)
        #expect(result.text == "Alpha is true [1].")
        #expect(result.sentences[0].label == .supported)
    }

    @Test func theScorerSeesTheClaimWithoutMarkersOrBullets() async throws {
        let scorer = ScriptedScorer([("Alpha", [0.9])])
        _ = try await AnswerVerifier.verify("- Alpha is true [1].", sources: sources, thresholds: defaults, scorer: scorer.scorer)
        #expect(scorer.recorded.first?.claim == "Alpha is true.")
    }

    @Test func eachListItemIsVerifiedOnItsOwnAndLayoutIsPreserved() async throws {
        let scorer = ScriptedScorer([("Alpha", [0.9]), ("Beta", [0.1])])
        let result = try await AnswerVerifier.verify("Facts:\n1. Alpha holds [1].\n2. Beta holds [1].\n\nDone.", sources: sources, thresholds: defaults, scorer: scorer.scorer)
        #expect(result.sentences.map(\.label) == [.unsupported, .supported, .unsupported, .unsupported], "“Facts:” and “Done.” carry no citation")
        // Both items cite source 1, so both keep [1]; the surrounding layout is untouched.
        #expect(result.text == "Facts:\n1. Alpha holds [1].\n2. Beta holds [1].\n\nDone.")
    }

    @Test func aMajorityOfUnsupportedSentencesIsReportedStrictly() async throws {
        let scorer = ScriptedScorer([("good", [0.9])])
        let twoOfThree = try await AnswerVerifier.verify("Bad one [1]. Bad two [1]. Good one [1].", sources: sources, thresholds: defaults, scorer: scorer.scorer)
        #expect(twoOfThree.mostlyUnsupported)
        let oneOfTwo = try await AnswerVerifier.verify("Bad one [1]. Good one [1].", sources: sources, thresholds: defaults, scorer: scorer.scorer)
        #expect(!oneOfTwo.mostlyUnsupported, "exactly half is not a majority")
    }

    @Test func anEmptyAnswerHasNothingSupportingIt() async throws {
        let scorer = ScriptedScorer([])
        for text in ["", "   \n ", "[1]"] {
            let result = try await AnswerVerifier.verify(text, sources: sources, thresholds: defaults, scorer: scorer.scorer)
            #expect(result.sentences.isEmpty && result.mostlyUnsupported && result.supportedFraction == 0, "“\(text)”")
        }
    }

    @Test func withNoSourcesNothingCanBeCited() async throws {
        let scorer = ScriptedScorer([("alpha", [0.9])])
        let result = try await AnswerVerifier.verify("Alpha [1].", sources: [], thresholds: defaults, scorer: scorer.scorer)
        #expect(result.sentences[0].label == .unsupported && scorer.recorded.isEmpty)
    }

    @Test func sentenceCallbacksArriveInOrderWithFinalNumbering() async throws {
        let scorer = ScriptedScorer([("Alpha", [0, 0.9]), ("Beta", [0.9, 0])])
        let seen = OSAllocatedUnfairLock(initialState: [VerifiedSentence]())
        let result = try await AnswerVerifier.verify(
            "Alpha [2]. Beta [1].", sources: sources, thresholds: defaults, scorer: scorer.scorer,
            onSentence: { sentence in seen.withLock { $0.append(sentence) } })
        #expect(seen.withLock { $0 } == result.sentences)
        #expect(seen.withLock { $0 }.map(\.citationIndices) == [[1], [2]])
    }

    @Test func scorerErrorsPropagate() async throws {
        struct Boom: Error {}
        await #expect(throws: Boom.self) {
            _ = try await AnswerVerifier.verify("Alpha [1].", sources: sources, thresholds: defaults) { _, _ in throw Boom() }
        }
    }

    @Test func cancellationStopsVerification() async throws {
        let task = Task {
            try await AnswerVerifier.verify(
                String(repeating: "A claim [1]. ", count: 200), sources: sources, thresholds: defaults
            ) { _, passages in
                try await Task.sleep(for: .milliseconds(20))
                return passages.map { _ in 0.9 }
            }
        }
        try await Task.sleep(for: .milliseconds(80))
        task.cancel()
        await #expect(throws: CancellationError.self) { _ = try await task.value }
    }
}
