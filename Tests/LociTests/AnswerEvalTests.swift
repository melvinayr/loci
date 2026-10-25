import Foundation
import Testing
@testable import Loci
@testable import LociEval
@testable import LociIngest
@testable import LociLlama

private let evalSmall = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Fixtures/eval-small", isDirectory: true)

private typealias Label = SupportExample.Label

@Suite("Verifier maths")
struct SupportMathTests {
    private let cutoffs = SupportCutoffs(supported: 0.6, weak: 0.35)

    @Test func labelsFallIntoBandsAtTheBoundaries() {
        #expect(cutoffs.label(for: 0.6) == .supported && cutoffs.label(for: 0.5999) == .weak)
        #expect(cutoffs.label(for: 0.35) == .weak && cutoffs.label(for: 0.3499) == .unsupported)
    }

    @Test func confusionMatrixAndPrecisionRecallMatchAHandComputation() {
        let examples: [SupportMath.Scored] = [
            (0.9, .supported), (0.8, .supported), (0.5, .weak),
            (0.7, .unsupported), (0.4, .unsupported), (0.1, .unsupported),
        ]
        let m = SupportMath.metrics(examples, cutoffs: cutoffs)
        // human supported → predicted supported ×2; weak → weak; unsupported → supported, weak, unsupported
        #expect(m.confusion == [[2, 0, 0], [0, 1, 0], [1, 1, 1]])
        #expect(abs(m.accuracy - 4.0 / 6) < 1e-12)
        #expect(m.unsupportedPrecision == 1 && abs(m.unsupportedRecall - 1.0 / 3) < 1e-12 && abs(m.unsupportedF1 - 0.5) < 1e-12)
        #expect(abs(m.supportedPrecision - 2.0 / 3) < 1e-12 && m.supportedRecall == 1 && abs(m.supportedF1 - 0.8) < 1e-12)
        #expect(m.examples == 6)
    }

    @Test func undefinedRatiosAreZeroNotNaN() {
        let none = SupportMath.metrics([], cutoffs: cutoffs)
        #expect(none.accuracy == 0 && none.unsupportedF1 == 0 && none.supportedPrecision == 0)
        let onlySupported = SupportMath.metrics([(0.9, .supported)], cutoffs: cutoffs)
        #expect(onlySupported.unsupportedPrecision == 0 && onlySupported.unsupportedRecall == 0 && onlySupported.unsupportedF1 == 0)
        #expect(onlySupported.supportedF1 == 1)
        for value in [none.unsupportedF1, onlySupported.unsupportedF1] { #expect(!value.isNaN) }
    }

    @Test func aucIsTheProbabilityAPositiveOutscoresANegative() {
        #expect(SupportMath.auc(positives: [0.9, 0.8], negatives: [0.7, 0.85]) == 0.75)
        #expect(SupportMath.auc(positives: [0.5], negatives: [0.5]) == 0.5, "ties count half")
        #expect(SupportMath.auc(positives: [0.9], negatives: [0.1]) == 1)
        #expect(SupportMath.auc(positives: [0.1], negatives: [0.9]) == 0)
        #expect(SupportMath.auc(positives: [], negatives: [0.9]) == 0.5)
    }

    @Test func calibrationSeparatesPerfectlySeparableLabels() {
        let examples: [SupportMath.Scored] = [
            (0.90, .supported), (0.95, .supported), (0.60, .weak), (0.65, .weak), (0.10, .unsupported), (0.20, .unsupported),
        ]
        let fitted = SupportMath.calibrate(examples)
        #expect(fitted.weak > 0.2 && fitted.weak <= 0.6, "between the unsupported and weak examples: \(fitted)")
        #expect(fitted.supported > 0.65 && fitted.supported <= 0.9, "between the weak and supported examples: \(fitted)")
        #expect(SupportMath.metrics(examples, cutoffs: fitted).accuracy == 1)
    }

    @Test func calibrationBeatsADefaultThatSitsInTheWrongPlace() {
        // Everything scores high, as with a reranker that calls anything on-topic relevant.
        let examples: [SupportMath.Scored] = [
            (0.999, .supported), (0.998, .supported), (0.997, .supported), (0.99, .weak),
            (0.97, .unsupported), (0.96, .unsupported), (0.998, .unsupported), (0.95, .unsupported),
        ]
        let atDefault = SupportMath.metrics(examples, cutoffs: cutoffs)
        let fitted = SupportMath.calibrate(examples)
        let atFitted = SupportMath.metrics(examples, cutoffs: fitted)
        #expect(atDefault.unsupportedF1 == 0)
        #expect(atFitted.unsupportedF1 > 0.7, "fitted \(fitted) gave F1 \(atFitted.unsupportedF1)")
        #expect(fitted.supported >= fitted.weak)
    }

    @Test func theFlagIsHeldToTheRequiredPrecisionAtTheCostOfRecall() {
        // Unsupported claims are scattered among supported ones: flagging more of them means flagging good ones too.
        let examples: [SupportMath.Scored] = [
            (0.05, .unsupported), (0.10, .unsupported), (0.30, .unsupported), (0.50, .supported), (0.55, .unsupported),
            (0.60, .supported), (0.70, .supported), (0.80, .unsupported), (0.90, .supported), (0.95, .supported),
        ]
        let strict = SupportMath.calibrate(examples, minimumPrecision: 1.0)
        let strictMetrics = SupportMath.metrics(examples, cutoffs: strict)
        #expect(strictMetrics.unsupportedPrecision == 1, "nothing good is flagged: \(strict)")
        #expect(strict.weak > 0.30 && strict.weak <= 0.50, "flags exactly the three clear cases: \(strict)")
        let loose = SupportMath.calibrate(examples, minimumPrecision: 0.6)
        let looseMetrics = SupportMath.metrics(examples, cutoffs: loose)
        #expect(looseMetrics.unsupportedRecall > strictMetrics.unsupportedRecall, "a lower precision floor buys recall")
        #expect(looseMetrics.unsupportedPrecision >= 0.6)
    }

    @Test func ifNoCutoffIsPreciseEnoughNothingIsFlagged() {
        // The lowest-scoring example is a good one, so any flag is wrong.
        let examples: [SupportMath.Scored] = [(0.1, .supported), (0.2, .unsupported), (0.9, .supported), (0.95, .unsupported)]
        let fitted = SupportMath.calibrate(examples, minimumPrecision: 0.95)
        #expect(examples.allSatisfy { fitted.label(for: $0.score) != .unsupported }, "\(fitted)")
    }

    @Test func theWeakBandCollapsesWhenTheScorerCannotSeparateWeakFromSupported() {
        // Weak and supported examples are interleaved (AUC 0.5), so no useful cut-off exists between them.
        let mixed: [SupportMath.Scored] = [
            (0.1, .unsupported), (0.15, .unsupported), (0.90, .supported), (0.96, .weak), (0.97, .supported), (0.91, .weak),
            (0.98, .supported), (0.95, .weak), (0.93, .supported), (0.99, .weak),
        ]
        let collapsed = SupportMath.calibrate(mixed)
        #expect(collapsed.supported == collapsed.weak, "\(collapsed)")
        #expect(SupportMath.metrics(mixed, cutoffs: collapsed).confusion.map { $0[1] } == [0, 0, 0], "no sentence is called weak")

        // When they do separate, the band stays.
        let separable: [SupportMath.Scored] = [
            (0.1, .unsupported), (0.15, .unsupported), (0.6, .weak), (0.65, .weak), (0.9, .supported), (0.95, .supported),
        ]
        let kept = SupportMath.calibrate(separable)
        #expect(kept.supported > kept.weak)
    }

    @Test func calibrationWithNothingToLearnFromReturnsTheDefaultsAndKeepsTheBandsOrdered() {
        #expect(SupportMath.calibrate([]) == SupportCutoffs(supported: 0.6, weak: 0.35))
        for examples: [SupportMath.Scored] in [[(0.5, .supported)], [(0.5, .unsupported)], [(0.2, .weak), (0.2, .weak)]] {
            let fitted = SupportMath.calibrate(examples)
            #expect(fitted.supported >= fitted.weak)
        }
    }

    @Test func meanOfNothingIsZero() {
        #expect(SupportMath.mean([]) == 0)
        #expect(abs(SupportMath.mean([0.2, 0.4]) - 0.3) < 1e-6)
    }
}

@Suite("Support examples dataset")
struct SupportDatasetTests {
    @Test func parsesExamplesAndRejectsBadOnes() throws {
        let line = #"{"id":"a","claim":"c","passage":"p","document":"d.md","label":"weak","split":"test"}"#
        let parsed = try Dataset.parseSupport(Data((line + "\n\n# comment\n").utf8))
        #expect(parsed == [SupportExample(id: "a", claim: "c", passage: "p", document: "d.md", label: .weak, split: .test)])
        #expect(throws: EvalError.self) { try Dataset.parseSupport(Data(#"{"id":"a","claim":"c","passage":"p","document":"d","label":"maybe","split":"test"}"#.utf8)) }
        #expect(throws: EvalError.self) { try Dataset.parseSupport(Data(#"{"id":"a","claim":"c","passage":"p","document":"d","label":"weak","split":"train"}"#.utf8)) }
        #expect(throws: EvalError.self) { try Dataset.parseSupport(Data("not json".utf8)) }
    }

    @Test func validationCatchesDuplicatesEmptyTextAndMissingDocuments() throws {
        let good = SupportExample(id: "a", claim: "c", passage: "p", document: "d.md", label: .supported, split: .test)
        #expect(throws: EvalError.self) { try Dataset.validate([good, good], documentsFolder: nil) }
        var empty = good; empty.id = "b"; empty.claim = "  "
        #expect(throws: EvalError.self) { try Dataset.validate([empty], documentsFolder: nil) }
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: EvalError.self) { try Dataset.validate([good], documentsFolder: directory) }
        try "x".write(to: directory.appendingPathComponent("d.md"), atomically: true, encoding: .utf8)
        try Dataset.validate([good], documentsFolder: directory)
    }

    @Test func theFixtureSubsetIsBalancedAndEveryPassageIsRealDocumentText() throws {
        let loaded = try Dataset.folder(evalSmall).load()
        let examples = loaded.supportExamples
        #expect(examples.count == 93)
        for label in Label.allCases { #expect(examples.filter { $0.label == label }.count >= 20, "\(label)") }
        for split in [SupportExample.Split.calibration, .test] {
            for label in Label.allCases {
                let count = examples.filter { $0.split == split && $0.label == label }.count
                let total = examples.filter { $0.label == label }.count
                #expect(abs(count - total / 2) <= 1, "\(split) \(label): \(count) of \(total)")
            }
        }
        let documents = try #require(loaded.documentsFolder)
        var texts: [String: String] = [:]
        for example in examples {
            if texts[example.document] == nil {
                texts[example.document] = Relevance.squash(try DocumentParser.parse(fileAt: documents.appendingPathComponent(example.document)).text)
            }
            #expect(texts[example.document]?.contains(Relevance.squash(example.passage)) == true, "\(example.id): passage is not in \(example.document)")
        }
        // Each claim is paired with exactly the passages it was written against: no claim repeats within a passage.
        #expect(Set(examples.map { $0.passage + "|" + $0.claim }).count == examples.count)
    }
}

private func withKnowledgeBase(
    generator: FakeGenerator? = nil, reranker: FakeReranker? = nil, _ body: (KnowledgeBase) async throws -> Void
) async throws {
    let directory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let services = await ModelServices(embedder: FakeEmbedder(), reranker: reranker, generator: generator)
    try await body(try await KnowledgeBase(directory: directory.appendingPathComponent("kb"), services: services))
}

private func overlapReranker() -> FakeReranker {
    FakeReranker { claim, passage in
        func words(_ t: String) -> Set<String> { Set(t.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init).filter { $0.count > 3 }) }
        return Float(words(claim).intersection(words(passage)).count) * 2 - 3
    }
}

@Suite("Verifier evaluation")
struct VerifierEvaluationTests {
    private func scored(_ rows: [(Float, Label, SupportExample.Split)]) -> [(example: SupportExample, score: Float)] {
        rows.enumerated().map { index, row in
            (SupportExample(id: "e\(index)", claim: "c", passage: "p", document: "d", label: row.1, split: row.2), row.0)
        }
    }

    @Test func reportsBothSplitsMeansAndAUC() {
        let data = scored([
            (0.9, .supported, .calibration), (0.1, .unsupported, .calibration), (0.5, .weak, .calibration),
            (0.8, .supported, .test), (0.3, .unsupported, .test), (0.6, .weak, .test),
        ])
        let report = VerifierEvaluation.report(data, scorer: "s", defaultCutoffs: SupportCutoffs(supported: 0.7, weak: 0.4), calibrate: false)
        #expect(report.examples == 6 && report.calibrated == nil)
        #expect(report.atDefault.calibration.examples == 3 && report.atDefault.test.examples == 3)
        #expect(abs((report.meanScore["supported"] ?? 0) - 0.85) < 1e-6 && abs((report.meanScore["unsupported"] ?? 0) - 0.2) < 1e-6)
        #expect(report.aucSupportedVsUnsupported == 1)
    }

    @Test func calibrationUsesOnlyTheCalibrationHalf() {
        // The calibration half is perfectly separable around 0.5; the test half is adversarial.
        let data = scored([
            (0.9, .supported, .calibration), (0.85, .supported, .calibration), (0.1, .unsupported, .calibration), (0.15, .unsupported, .calibration),
            (0.2, .supported, .test), (0.95, .unsupported, .test),
        ])
        let report = VerifierEvaluation.report(data, scorer: "s", defaultCutoffs: SupportCutoffs(supported: 0.6, weak: 0.35), calibrate: true)
        let calibrated = try! #require(report.calibrated)
        #expect(calibrated.cutoffs.weak > 0.15 && calibrated.cutoffs.weak <= 0.85, "fitted to the calibration half only: \(calibrated.cutoffs)")
        #expect(calibrated.calibration.accuracy >= 0.5 && calibrated.calibration.unsupportedF1 == 1)
        #expect(calibrated.test.unsupportedF1 == 0, "the test half is reported, not fitted")
    }

    @Test func theEvaluatorScoresTheFixtureSubsetThroughTheKnowledgeBase() async throws {
        try await withKnowledgeBase(reranker: overlapReranker()) { kb in
            let report = try await Evaluator(knowledgeBase: kb)
                .run(dataset: .folder(evalSmall), sections: .verifier, verifier: VerifierOptions(calibrate: true))
            let verifier = try #require(report.verifier)
            #expect(report.retrieval.isEmpty && report.answers == nil)
            #expect(verifier.examples == 93 && verifier.scorer.contains("overlap") == false || verifier.scorer.hasPrefix("reranker"))
            #expect(verifier.calibrated != nil)
            #expect((0...1).contains(verifier.aucSupportedVsUnsupported))
            #expect(verifier.atDefault.calibration.examples + verifier.atDefault.test.examples == 93)
            let markdown = report.markdown()
            #expect(markdown.contains("Verification against hand labels") && markdown.contains("calibrated") && markdown.contains("Test-half confusion"))
        }
    }

    @Test func withoutARerankerTheEmbeddingFallbackIsNamed() async throws {
        try await withKnowledgeBase { kb in
            let report = try await Evaluator(knowledgeBase: kb).run(dataset: .folder(evalSmall), sections: .verifier)
            #expect(report.verifier?.scorer.hasPrefix("embedding similarity") == true)
        }
    }

    @Test func sectionsThatCannotRunSayWhy() async throws {
        try await withKnowledgeBase { kb in
            await #expect(throws: EvalError.self) { try await Evaluator(knowledgeBase: kb).run(dataset: .folder(evalSmall), sections: .answers) }
            let directory = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            try FileManager.default.createDirectory(at: directory.appendingPathComponent("documents"), withIntermediateDirectories: true)
            try "x".write(to: directory.appendingPathComponent("documents/a.txt"), atomically: true, encoding: .utf8)
            try #"{"id":"q","question":"?","relevant":[{"document":"a.txt"}]}"#.write(to: directory.appendingPathComponent("questions.jsonl"), atomically: true, encoding: .utf8)
            await #expect(throws: EvalError.self) { try await Evaluator(knowledgeBase: kb).run(dataset: .folder(directory), sections: .verifier) }
        }
    }
}

@Suite("Answer evaluation")
struct AnswerEvaluationTests {
    /// Two documents and three questions: two answerable, one not.
    private func makeDataset(in directory: URL) throws -> URL {
        let root = directory.appendingPathComponent("tiny")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("documents"), withIntermediateDirectories: true)
        try "The Cape lighthouse tower is 38 metres tall and was built in 1874."
            .write(to: root.appendingPathComponent("documents/lighthouse.txt"), atomically: true, encoding: .utf8)
        try "Feed the sourdough starter once a day and bake the loaf at 245 degrees."
            .write(to: root.appendingPathComponent("documents/bread.txt"), atomically: true, encoding: .utf8)
        let questions = [
            #"{"id":"q1","question":"How tall is the lighthouse tower?","relevant":[{"document":"lighthouse.txt","snippet":"38 metres"}],"answer":"38 metres"}"#,
            #"{"id":"q2","question":"At what temperature is the loaf baked?","relevant":[{"document":"bread.txt","snippet":"245 degrees"}],"answer":"245 degrees"}"#,
            #"{"id":"q3","question":"Who is the mayor of Aldermere?","answerable":false}"#,
        ]
        try questions.joined(separator: "\n").write(to: root.appendingPathComponent("questions.jsonl"), atomically: true, encoding: .utf8)
        return root
    }

    private func generator(_ replies: [String: String]) -> FakeGenerator {
        FakeGenerator { messages in
            let user = messages.last?.content ?? ""
            return replies.first { user.contains($0.key) }?.value ?? "I couldn't find the answer in your documents."
        }
    }

    private let options = AnswerOptions(retrieval: RetrievalOptions(limit: 2, mode: .keyword, rerank: false), temperature: 0)

    @Test func scoresCitationsFaithfulnessRefusalsAndAnswerMatch() async throws {
        let replies = [
            "How tall is the lighthouse tower?": "The lighthouse tower is 38 metres tall [1].",
            "At what temperature is the loaf baked?": "The loaf is baked at 245 degrees [1]. It also needs butter on top [1].",
        ]
        try await withKnowledgeBase(generator: generator(replies), reranker: overlapReranker()) { kb in
            let directory = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let report = try await Evaluator(knowledgeBase: kb, modes: [.keyword])
                .run(dataset: .folder(makeDataset(in: directory)), sections: .answers, answerOptions: options)
            let answers = try #require(report.answers)

            #expect(answers.answerable == 2 && answers.unanswerable == 1 && answers.rows.count == 3)
            #expect(answers.citations.count == 2, "q1 cites one source; q2's two sentences cite the same one: \(answers.rows.map(\.citations))")
            #expect(answers.citations.precision == 1 && answers.citations.recall == 1)
            #expect(answers.citations.structuralValidity == 1 && answers.citations.checked >= 2)
            #expect(answers.refusal.accuracy == 1 && answers.refusal.falseRefusalRate == 0)
            #expect(answers.answerMatch.withReference == 2 && answers.answerMatch.fraction == 1)
            #expect(answers.faithfulness.sentences == 3)
            let supported = answers.rows.reduce(0) { $0 + $1.supported }, unsupported = answers.rows.reduce(0) { $0 + $1.unsupported }
            #expect(supported == 2 && unsupported == 1, "the made-up butter sentence is the unsupported one")
            #expect(abs(answers.faithfulness.unsupportedRate - 1.0 / 3) < 1e-9)
            #expect(answers.performance.totalP50Ms > 0 && answers.performance.indexBytes > 0)
            #expect(answers.settings.promptVersion == KnowledgeBase.groundingPromptVersion)
            #expect(report.environment.generator == "test-generator")
        }
    }

    @Test func aCitationToTheWrongDocumentLowersPrecisionAndAFalseRefusalIsCounted() async throws {
        let replies = [
            // Sources are ranked by keyword search: this answer cites the bread passage for a lighthouse question.
            "How tall is the lighthouse tower?": "The lighthouse tower is 38 metres tall [2].",
        ]
        try await withKnowledgeBase(generator: generator(replies), reranker: overlapReranker()) { kb in
            let directory = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            var lenient = options
            lenient.refuseWhenUnsupported = false
            let report = try await Evaluator(knowledgeBase: kb, modes: [.keyword])
                .run(dataset: .folder(makeDataset(in: directory)), sections: .answers, answerOptions: lenient)
            let answers = try #require(report.answers)
            let q1 = try #require(answers.rows.first { $0.id == "q1" })
            // q2 has no scripted reply, so the fake model refuses it: a false refusal.
            #expect(answers.refusal.falseRefusalRate == 0.5 && answers.refusal.accuracy == 1)
            #expect(q1.citations <= 1 && answers.citations.recall <= 0.5)
            #expect(answers.answerMatch.fraction <= 0.5)
        }
    }

    @Test func theMarkdownReportListsEverySectionAndTheRegressionRulesApply() async throws {
        let replies = ["How tall is the lighthouse tower?": "The lighthouse tower is 38 metres tall [1]."]
        try await withKnowledgeBase(generator: generator(replies), reranker: overlapReranker()) { kb in
            let directory = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let report = try await Evaluator(knowledgeBase: kb, modes: [.keyword])
                .run(dataset: .folder(makeDataset(in: directory)), sections: [.retrieval, .answers], answerOptions: options)
            let markdown = report.markdown()
            for heading in ["## Retrieval", "## Answers", "### Citations", "### Faithfulness", "### Refusals and answer match", "### Performance"] {
                #expect(markdown.contains(heading), Comment(rawValue: heading))
            }
            #expect(markdown.contains("Prompt v\(KnowledgeBase.groundingPromptVersion)") && markdown.contains("Structural validity"))

            // JSON round trip keeps the new sections.
            let url = directory.appendingPathComponent("r.json")
            try report.write(to: url, format: .json)
            let back = try EvalReport.read(from: url)
            #expect(back.answers == report.answers && back.retrieval == report.retrieval)

            // Regression policy.
            #expect(report.regressions(against: report).isEmpty)
            var worse = report
            worse.answers?.citations.precision = max((report.answers?.citations.precision ?? 1) - 0.1, 0)
            #expect(worse.regressions(against: report).contains { $0.contains("Citation precision") })
            var invalid = report
            invalid.answers?.citations.structuralValidity = 0.99
            #expect(invalid.regressions(against: report).contains { $0.contains("Structural validity") })
            #expect(invalid.regressions(against: EvalReportTesting.withoutAnswers(report)).contains { $0.contains("Structural validity") },
                    "structural validity is checked even without a baseline for answers")
        }
    }

    @Test func thresholdsFittedByCalibrationAreUsedForTheAnswers() async throws {
        let replies = ["How tall is the lighthouse tower?": "The lighthouse tower is 38 metres tall [1]."]
        try await withKnowledgeBase(generator: generator(replies), reranker: overlapReranker()) { kb in
            let directory = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            // Needs a support file for calibration.
            let root = try makeDataset(in: directory)
            let lines = try (0..<6).map { index -> String in
                let example = SupportExample(
                    id: "s\(index)", claim: index % 2 == 0 ? "The lighthouse tower is 38 metres tall." : "Penguins migrate every winter.",
                    passage: "The Cape lighthouse tower is 38 metres tall and was built in 1874.", document: "lighthouse.txt",
                    label: index % 2 == 0 ? .supported : .unsupported, split: index < 4 ? .calibration : .test)
                return String(decoding: try JSONEncoder().encode(example), as: UTF8.self)
            }
            try lines.joined(separator: "\n").write(to: root.appendingPathComponent("support.jsonl"), atomically: true, encoding: .utf8)

            let report = try await Evaluator(knowledgeBase: kb, modes: [.keyword])
                .run(dataset: .folder(root), sections: [.answers, .verifier], verifier: VerifierOptions(calibrate: true))
            let fitted = try #require(report.verifier?.calibrated?.cutoffs)
            let used = try #require(report.answers?.settings)
            #expect(used.supportedThreshold == fitted.supported && used.weakThreshold == fitted.weak)
        }
    }
}

private enum EvalReportTesting {
    static func withoutAnswers(_ report: EvalReport) -> EvalReport {
        var copy = report
        copy.answers = nil
        return copy
    }
}
