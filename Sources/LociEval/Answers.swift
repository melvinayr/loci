import Foundation
import Loci

/// What happened for one question.
public struct AnswerRow: Codable, Sendable, Equatable {
    public var id: String
    public var answerable: Bool
    public var refused: Bool
    public var sentences: Int
    public var supported: Int
    public var weak: Int
    public var unsupported: Int
    public var citations: Int
    /// Citations that point at a ground-truth passage.
    public var relevantCitations: Int
    /// Whether the answer contains the reference answer; nil when the question has none.
    public var answerMatch: Bool?
    public var seconds: Double
    /// The support score of each sentence, in order (empty for a refusal or when verification was off).
    public var sentenceScores: [Float]
}

public struct AnswerReport: Codable, Sendable, Equatable {
    public struct Settings: Codable, Sendable, Equatable {
        public var promptVersion: Int
        public var retrievalLimit: Int
        public var rerank: Bool
        public var maxAnswerTokens: Int
        public var temperature: Float
        public var verify: Bool
        public var supportedThreshold: Float
        public var weakThreshold: Float
        public var refuseWhenUnsupported: Bool
    }

    public struct Citations: Codable, Sendable, Equatable {
        /// Citations in answers to answerable questions that were not refused.
        public var count: Int
        /// Of those, the fraction pointing at a ground-truth passage.
        public var precision: Double
        /// Fraction of answerable questions whose answer has at least one citation to a ground-truth passage.
        public var recall: Double
        /// Fraction of all citations (refusals' included) whose range slices to the excerpt, and for PDFs whose page contains the range.
        /// Anything below 1 is a bug.
        public var structuralValidity: Double
        public var checked: Int
    }

    public struct Faithfulness: Codable, Sendable, Equatable {
        /// Sentences in answers that were not refused.
        public var sentences: Int
        public var supportedFraction: Double
        public var weakFraction: Double
        public var unsupportedRate: Double
    }

    public struct Refusal: Codable, Sendable, Equatable {
        /// Unanswerable questions that were refused, over all unanswerable questions.
        public var accuracy: Double
        /// Answerable questions that were refused, over all answerable questions.
        public var falseRefusalRate: Double
        public var unanswerable: Int
        public var answerable: Int
    }

    public struct AnswerMatch: Codable, Sendable, Equatable {
        /// Answerable questions with a reference answer whose answer contains it (ignoring case and punctuation).
        public var fraction: Double
        public var withReference: Int
    }

    public struct Performance: Codable, Sendable, Equatable {
        public var retrievalP50Ms: Double
        public var rerankP50Ms: Double
        public var timeToFirstTokenP50Ms: Double
        public var timeToFirstTokenP95Ms: Double
        public var verificationP50Ms: Double
        public var totalP50Ms: Double
        public var totalP95Ms: Double
        public var tokensPerSecondMean: Double
        public var peakMemoryBytes: UInt64
        public var indexBytes: UInt64
    }

    public var settings: Settings
    public var answerable: Int
    public var unanswerable: Int
    public var citations: Citations
    public var faithfulness: Faithfulness
    public var refusal: Refusal
    public var answerMatch: AnswerMatch
    public var performance: Performance
    public var rows: [AnswerRow]
}

enum AnswerEvaluation {
    static func run(
        _ questions: [EvalQuestion], knowledgeBase: KnowledgeBase, options: AnswerOptions
    ) async throws -> AnswerReport {
        var rows: [AnswerRow] = []
        var timings: [Double] = [], retrieval: [Double] = [], rerank: [Double] = [], firstToken: [Double] = []
        var verification: [Double] = [], speeds: [Double] = []
        var peak: UInt64 = 0
        var structuralChecked = 0, structuralValid = 0

        // Load every model before timing anything.
        if let first = questions.first { _ = try await knowledgeBase.ask(first.question, options: options) }

        for question in questions {
            let start = ContinuousClock.now
            let answer = try await knowledgeBase.ask(question.question, options: options)
            let seconds = seconds(ContinuousClock.now - start)

            var relevantCitations = 0
            for citation in answer.citations {
                if question.relevant.contains(where: { Relevance.matches(document: citation.document, page: citation.page, text: citation.excerpt, $0) }) {
                    relevantCitations += 1
                }
                structuralChecked += 1
                if try await isStructurallyValid(citation, knowledgeBase) { structuralValid += 1 }
            }
            let match: Bool? = question.answerable ? question.answer.map {
                !answer.refused && Relevance.normalize(answer.text).contains(Relevance.normalize($0))
            } : nil

            rows.append(AnswerRow(
                id: question.id, answerable: question.answerable, refused: answer.refused, sentences: answer.sentences.count,
                supported: answer.sentences.filter { $0.support == .supported }.count,
                weak: answer.sentences.filter { $0.support == .weak }.count,
                unsupported: answer.sentences.filter { $0.support == .unsupported }.count,
                citations: answer.refused ? 0 : answer.citations.count,
                relevantCitations: answer.refused ? 0 : relevantCitations, answerMatch: match, seconds: seconds,
                sentenceScores: answer.sentences.map(\.score)))

            timings.append(seconds * 1000)
            retrieval.append(ms(answer.stats.retrievalTime))
            rerank.append(ms(answer.stats.rerankTime))
            if !answer.refused || answer.stats.timeToFirstToken > .zero { firstToken.append(ms(answer.stats.timeToFirstToken)) }
            if answer.stats.verificationTime > .zero { verification.append(ms(answer.stats.verificationTime)) }
            if answer.stats.generationTokensPerSecond > 0 { speeds.append(answer.stats.generationTokensPerSecond) }
            peak = max(peak, answer.stats.peakMemoryBytes)
        }

        let answerable = rows.filter(\.answerable), unanswerable = rows.filter { !$0.answerable }
        let answered = rows.filter { !$0.refused }
        let citationCount = answerable.filter { !$0.refused }.reduce(0) { $0 + $1.citations }
        let relevant = answerable.reduce(0) { $0 + $1.relevantCitations }
        let sentences = answered.reduce(0) { $0 + $1.sentences }
        func fraction(_ count: Int, of total: Int) -> Double { total == 0 ? 0 : Double(count) / Double(total) }
        let references = answerable.compactMap(\.answerMatch)

        return AnswerReport(
            settings: .init(
                promptVersion: KnowledgeBase.groundingPromptVersion, retrievalLimit: options.retrieval.limit,
                rerank: options.retrieval.rerank, maxAnswerTokens: options.maxAnswerTokens, temperature: options.temperature,
                verify: options.verify, supportedThreshold: options.supportThresholds.supported,
                weakThreshold: options.supportThresholds.weak, refuseWhenUnsupported: options.refuseWhenUnsupported),
            answerable: answerable.count, unanswerable: unanswerable.count,
            citations: .init(
                count: citationCount, precision: fraction(relevant, of: citationCount),
                recall: fraction(answerable.filter { $0.relevantCitations > 0 }.count, of: answerable.count),
                structuralValidity: structuralChecked == 0 ? 1 : fraction(structuralValid, of: structuralChecked), checked: structuralChecked),
            faithfulness: .init(
                sentences: sentences, supportedFraction: fraction(answered.reduce(0) { $0 + $1.supported }, of: sentences),
                weakFraction: fraction(answered.reduce(0) { $0 + $1.weak }, of: sentences),
                unsupportedRate: fraction(answered.reduce(0) { $0 + $1.unsupported }, of: sentences)),
            refusal: .init(
                accuracy: fraction(unanswerable.filter(\.refused).count, of: unanswerable.count),
                falseRefusalRate: fraction(answerable.filter(\.refused).count, of: answerable.count),
                unanswerable: unanswerable.count, answerable: answerable.count),
            answerMatch: .init(fraction: fraction(references.filter { $0 }.count, of: references.count), withReference: references.count),
            performance: .init(
                retrievalP50Ms: Metrics.percentile(retrieval, 0.5), rerankP50Ms: Metrics.percentile(rerank, 0.5),
                timeToFirstTokenP50Ms: Metrics.percentile(firstToken, 0.5), timeToFirstTokenP95Ms: Metrics.percentile(firstToken, 0.95),
                verificationP50Ms: Metrics.percentile(verification, 0.5), totalP50Ms: Metrics.percentile(timings, 0.5),
                totalP95Ms: Metrics.percentile(timings, 0.95), tokensPerSecondMean: Metrics.mean(speeds), peakMemoryBytes: peak,
                indexBytes: await knowledgeBase.stats.databaseBytes),
            rows: rows)
    }

    /// The citation's range slices to exactly its excerpt in the stored text and, for a PDF, lies inside its page.
    private static func isStructurallyValid(_ citation: Citation, _ knowledgeBase: KnowledgeBase) async throws -> Bool {
        guard let text = try await knowledgeBase.documentText(for: citation.document.id) else { return false }
        let scalars = Array(text.unicodeScalars)
        guard citation.range.lowerBound >= 0, citation.range.upperBound <= scalars.count, !citation.range.isEmpty else { return false }
        guard String(String.UnicodeScalarView(scalars[citation.range])) == citation.excerpt else { return false }
        if citation.document.kind == .pdf {
            guard let page = citation.page, let ranges = try await knowledgeBase.pageRanges(for: citation.document.id),
                  ranges.indices.contains(page - 1), let pageRange = ranges[page - 1] else { return false }
            return pageRange.lowerBound <= citation.range.lowerBound && citation.range.upperBound <= pageRange.upperBound
        }
        return citation.page == nil
    }

    private static func ms(_ duration: Duration) -> Double { seconds(duration) * 1000 }

    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}
