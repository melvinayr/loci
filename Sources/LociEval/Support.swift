import Foundation
import Loci

/// A hand-labelled (claim, passage) pair for judging answer verification.
public struct SupportExample: Codable, Sendable, Equatable {
    public enum Label: String, Codable, Sendable, CaseIterable { case supported, weak, unsupported }
    public enum Split: String, Codable, Sendable { case calibration, test }

    public var id: String
    /// A sentence such as a model might write in an answer.
    public var claim: String
    /// Text from `document`: the passage the claim cites.
    public var passage: String
    public var document: String
    public var label: Label
    /// Thresholds are fitted on `calibration` and checked on `test`.
    public var split: Split

    public init(id: String, claim: String, passage: String, document: String, label: Label, split: Split) {
        self.id = id
        self.claim = claim
        self.passage = passage
        self.document = document
        self.label = label
        self.split = split
    }
}

/// Support-score cut-offs on a 0–1 scale, as in `AnswerOptions.supportThresholds`.
public struct SupportCutoffs: Codable, Sendable, Equatable {
    public var supported: Float
    public var weak: Float

    public init(supported: Float, weak: Float) {
        self.supported = supported
        self.weak = weak
    }

    func label(for score: Float) -> SupportExample.Label {
        score >= supported ? .supported : score >= weak ? .weak : .unsupported
    }
}

/// How well verification's labels match the human ones on one set of examples.
public struct VerifierMetrics: Codable, Sendable, Equatable {
    public var examples: Int
    /// Fraction labelled exactly as the human did (three classes).
    public var accuracy: Double
    /// For the `unsupported` flag: of the sentences flagged, how many really were unsupported; and of the unsupported
    /// ones, how many were flagged.
    public var unsupportedPrecision: Double
    public var unsupportedRecall: Double
    public var unsupportedF1: Double
    public var supportedPrecision: Double
    public var supportedRecall: Double
    public var supportedF1: Double
    /// `confusion[human][predicted]`, both ordered supported, weak, unsupported.
    public var confusion: [[Int]]
}

/// Cut-offs fitted to the calibration examples, and how they perform.
public struct CalibratedCutoffs: Codable, Sendable, Equatable {
    public var cutoffs: SupportCutoffs
    public var calibration: VerifierMetrics
    public var test: VerifierMetrics
}

public struct VerifierReport: Codable, Sendable, Equatable {
    /// What produced the scores: the reranker's ID, or the embedder's when there is none.
    public var scorer: String
    public var examples: Int
    /// Mean score per human label, a quick read on how well the scorer separates them.
    public var meanScore: [String: Double]
    /// Probability that a randomly chosen unsupported example scores below a randomly chosen supported one (0.5 is chance).
    public var aucSupportedVsUnsupported: Double
    /// Probability that a supported example outscores a weak one. Near 0.5 means the scorer cannot tell them apart.
    public var aucSupportedVsWeak: Double
    /// The precision the fitted `weak` cut-off was held to.
    public var minimumPrecision: Double
    public var defaultCutoffs: SupportCutoffs
    public var atDefault: SplitMetrics
    public var calibrated: CalibratedCutoffs?
    /// The score every example received, for analysis beyond the summary numbers.
    public var scores: [ScoredExample]

    public struct ScoredExample: Codable, Sendable, Equatable {
        public var id: String
        public var label: SupportExample.Label
        public var split: SupportExample.Split
        public var score: Float
    }

    public struct SplitMetrics: Codable, Sendable, Equatable {
        public var calibration: VerifierMetrics
        public var test: VerifierMetrics
    }
}

/// The arithmetic behind the verifier metrics.
enum SupportMath {
    typealias Scored = (score: Float, label: SupportExample.Label)

    static func metrics(_ examples: [Scored], cutoffs: SupportCutoffs) -> VerifierMetrics {
        let order = SupportExample.Label.allCases  // supported, weak, unsupported
        var confusion = [[Int]](repeating: [Int](repeating: 0, count: 3), count: 3)
        for example in examples {
            let human = order.firstIndex(of: example.label) ?? 0
            let predicted = order.firstIndex(of: cutoffs.label(for: example.score)) ?? 0
            confusion[human][predicted] += 1
        }
        let correct = (0..<3).reduce(0) { $0 + confusion[$1][$1] }
        func prf(_ index: Int) -> (precision: Double, recall: Double, f1: Double) {
            let truePositive = confusion[index][index]
            let predicted = (0..<3).reduce(0) { $0 + confusion[$1][index] }
            let actual = confusion[index].reduce(0, +)
            return f1(truePositive: truePositive, predicted: predicted, actual: actual)
        }
        let supported = prf(0), unsupported = prf(2)
        return VerifierMetrics(
            examples: examples.count, accuracy: examples.isEmpty ? 0 : Double(correct) / Double(examples.count),
            unsupportedPrecision: unsupported.precision, unsupportedRecall: unsupported.recall, unsupportedF1: unsupported.f1,
            supportedPrecision: supported.precision, supportedRecall: supported.recall, supportedF1: supported.f1,
            confusion: confusion)
    }

    static func f1(truePositive: Int, predicted: Int, actual: Int) -> (precision: Double, recall: Double, f1: Double) {
        let precision = predicted == 0 ? 0 : Double(truePositive) / Double(predicted)
        let recall = actual == 0 ? 0 : Double(truePositive) / Double(actual)
        let f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall)
        return (precision, recall, f1)
    }

    /// Below this, a scorer cannot tell `weak` examples from `supported` ones, and the weak band is collapsed.
    static let minimumWeakSeparation = 0.7

    /// Cut-offs for a scorer, fitted to labelled examples.
    ///
    /// `weak` is the cut-off that flags the most unsupported examples while at least `minimumPrecision` of what it flags really is
    /// unsupported (ties take the middle of the tied range); if no cut-off reaches that precision it flags nothing. Maximising F1
    /// instead was tried and rejected: it set the cut-off so high that 73% of correct answers were refused.
    ///
    /// `supported` is chosen the same way for recognising supported examples, but only if the scorer separates `supported` from
    /// `weak` (AUC at least `minimumWeakSeparation`); otherwise it equals `weak`, so no sentence is called weak on the strength of a
    /// distinction the scorer cannot make.
    static func calibrate(_ examples: [Scored], minimumPrecision: Double = 0.95) -> SupportCutoffs {
        let scores = Set(examples.map(\.score)).sorted()
        guard let low = scores.first, let high = scores.last else { return SupportCutoffs(supported: 0.6, weak: 0.35) }
        // Candidate cut-offs sit between neighbouring distinct scores (and just outside the ends).
        var candidates: [Float] = [low - 0.001]
        candidates += zip(scores, scores.dropFirst()).map { ($0 + $1) / 2 }
        candidates.append(high + 0.001)

        let unsupportedTotal = examples.filter { $0.label == .unsupported }.count
        let weakCutoff = bestRecall(candidates, minimumPrecision: minimumPrecision) { threshold in
            let flagged = examples.filter { $0.score < threshold }
            return (flagged.filter { $0.label == .unsupported }.count, flagged.count, unsupportedTotal)
        } ?? low - 0.001

        let supportedScores = examples.filter { $0.label == .supported }.map(\.score)
        let weakScores = examples.filter { $0.label == .weak }.map(\.score)
        guard !weakScores.isEmpty, !supportedScores.isEmpty,
              auc(positives: supportedScores, negatives: weakScores) >= minimumWeakSeparation else {
            return SupportCutoffs(supported: weakCutoff, weak: weakCutoff)
        }
        let supportedTotal = supportedScores.count
        let supportedCutoff = bestRecall(candidates.filter { $0 >= weakCutoff }, minimumPrecision: minimumPrecision) { threshold in
            let predicted = examples.filter { $0.score >= threshold }
            return (predicted.filter { $0.label == .supported }.count, predicted.count, supportedTotal)
        }
        return SupportCutoffs(supported: supportedCutoff ?? weakCutoff, weak: weakCutoff)
    }

    /// Among candidates whose precision reaches `minimumPrecision`, the one with the best recall (middle of ties); nil if none does.
    private static func bestRecall(
        _ candidates: [Float], minimumPrecision: Double, counts: (Float) -> (truePositive: Int, predicted: Int, actual: Int)
    ) -> Float? {
        var scored: [(candidate: Float, recall: Double)] = []
        for candidate in candidates {
            let c = counts(candidate)
            guard c.predicted > 0 else {
                scored.append((candidate, 0))  // flags nothing: trivially precise, zero recall
                continue
            }
            let precision = Double(c.truePositive) / Double(c.predicted)
            if precision >= minimumPrecision { scored.append((candidate, c.actual == 0 ? 0 : Double(c.truePositive) / Double(c.actual))) }
        }
        guard let top = scored.map(\.recall).max() else { return nil }
        let tied = scored.filter { abs($0.recall - top) < 1e-12 }.map(\.candidate)
        return tied[tied.count / 2]
    }

    /// Probability that a random positive scores above a random negative (ties count half).
    static func auc(positives: [Float], negatives: [Float]) -> Double {
        guard !positives.isEmpty, !negatives.isEmpty else { return 0.5 }
        var wins = 0.0
        for p in positives {
            for n in negatives { wins += p > n ? 1 : p == n ? 0.5 : 0 }
        }
        return wins / Double(positives.count * negatives.count)
    }

    static func mean(_ values: [Float]) -> Double {
        values.isEmpty ? 0 : values.reduce(0) { $0 + Double($1) } / Double(values.count)
    }
}

/// Scores the labelled examples with a knowledge base's verifier and summarises agreement with the human labels.
enum VerifierEvaluation {
    static func run(
        _ examples: [SupportExample], knowledgeBase: KnowledgeBase, defaultCutoffs: SupportCutoffs, calibrate: Bool,
        minimumPrecision: Double = 0.95
    ) async throws -> VerifierReport {
        var scored: [(example: SupportExample, score: Float)] = []
        for example in examples {
            let scores = try await knowledgeBase.supportScores(for: example.claim, against: [example.passage])
            scored.append((example, scores.first ?? 0))
        }
        return report(
            scored, scorer: await scorerName(knowledgeBase), defaultCutoffs: defaultCutoffs, calibrate: calibrate,
            minimumPrecision: minimumPrecision)
    }

    static func report(
        _ scored: [(example: SupportExample, score: Float)], scorer: String, defaultCutoffs: SupportCutoffs, calibrate: Bool,
        minimumPrecision: Double = 0.95
    ) -> VerifierReport {
        func points(_ split: SupportExample.Split) -> [SupportMath.Scored] {
            scored.filter { $0.example.split == split }.map { ($0.score, $0.example.label) }
        }
        func splits(_ cutoffs: SupportCutoffs) -> VerifierReport.SplitMetrics {
            .init(
                calibration: SupportMath.metrics(points(.calibration), cutoffs: cutoffs),
                test: SupportMath.metrics(points(.test), cutoffs: cutoffs))
        }
        func scores(_ label: SupportExample.Label) -> [Float] { scored.filter { $0.example.label == label }.map(\.score) }

        var calibrated: CalibratedCutoffs?
        if calibrate {
            let fitted = SupportMath.calibrate(points(.calibration), minimumPrecision: minimumPrecision)
            let result = splits(fitted)
            calibrated = CalibratedCutoffs(cutoffs: fitted, calibration: result.calibration, test: result.test)
        }
        return VerifierReport(
            scorer: scorer, examples: scored.count,
            meanScore: Dictionary(uniqueKeysWithValues: SupportExample.Label.allCases.map { ($0.rawValue, SupportMath.mean(scores($0))) }),
            aucSupportedVsUnsupported: SupportMath.auc(positives: scores(.supported), negatives: scores(.unsupported)),
            aucSupportedVsWeak: SupportMath.auc(positives: scores(.supported), negatives: scores(.weak)),
            minimumPrecision: minimumPrecision,
            defaultCutoffs: defaultCutoffs, atDefault: splits(defaultCutoffs), calibrated: calibrated,
            scores: scored.map { .init(id: $0.example.id, label: $0.example.label, split: $0.example.split, score: $0.score) })
    }

    private static func scorerName(_ knowledgeBase: KnowledgeBase) async -> String {
        let stats = await knowledgeBase.stats
        if let reranker = stats.rerankerID { return "reranker \(reranker)" }
        return "embedding similarity (\(stats.embedderID ?? "unknown embedder"))"
    }
}
