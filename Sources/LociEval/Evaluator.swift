import Foundation
import Loci

/// A retrieval configuration to evaluate.
public struct EvalMode: Sendable, Hashable {
    /// The name used in reports and baselines.
    public let name: String
    public let retrieval: RetrievalMode
    public let rerank: Bool

    public static let keyword = EvalMode(name: "keyword", retrieval: .keyword, rerank: false)
    public static let vector = EvalMode(name: "vector", retrieval: .vector, rerank: false)
    public static let hybrid = EvalMode(name: "hybrid", retrieval: .hybrid, rerank: false)
    /// Hybrid retrieval followed by the cross-encoder; needs a reranker in the knowledge base's `ModelSet`.
    public static let hybridRerank = EvalMode(name: "hybrid+rerank", retrieval: .hybrid, rerank: true)

    /// Parses a report name such as `"hybrid+rerank"`.
    public init?(name: String) {
        switch name {
        case "keyword": self = .keyword
        case "vector": self = .vector
        case "hybrid": self = .hybrid
        case "hybrid+rerank": self = .hybridRerank
        default: return nil
        }
    }

    private init(name: String, retrieval: RetrievalMode, rerank: Bool) {
        self.name = name
        self.retrieval = retrieval
        self.rerank = rerank
    }
}

/// What an evaluation measures.
public struct EvalSections: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    /// Recall, MRR and nDCG per retrieval mode.
    public static let retrieval = EvalSections(rawValue: 1)
    /// Full answers: citation precision and recall, faithfulness, refusals, answer match and timings. Needs a generator.
    public static let answers = EvalSections(rawValue: 2)
    /// How well verification's labels match the hand-labelled `support.jsonl`. Needs that file in the dataset.
    public static let verifier = EvalSections(rawValue: 4)
}

/// Settings for the verifier section.
public struct VerifierOptions: Sendable {
    /// The cut-offs to report agreement for.
    public var cutoffs: SupportCutoffs
    /// Also fit cut-offs on the calibration examples and report how they do on both halves. When the answers
    /// section runs too, it uses the fitted cut-offs.
    public var calibrate: Bool
    /// How precise the fitted `weak` cut-off must be: of the sentences it flags as unsupported, at least this fraction really are.
    public var minimumPrecision: Double

    public init(cutoffs: SupportCutoffs = SupportCutoffs(supported: 0.6, weak: 0.35), calibrate: Bool = false, minimumPrecision: Double = 0.95) {
        self.cutoffs = cutoffs
        self.calibrate = calibrate
        self.minimumPrecision = minimumPrecision
    }
}

/// Measures how well a knowledge base answers a dataset's questions.
public struct Evaluator: Sendable {
    private let knowledgeBase: KnowledgeBase
    private let requestedModes: [EvalMode]?
    private let ingestOverride: IngestOptions?

    /// - Parameters:
    ///   - modes: what to compare. By default keyword, vector and hybrid, plus hybrid+rerank when the
    ///     knowledge base has a reranker.
    ///   - ingestOptions: how a `.folder` dataset's documents are chunked. By default the dataset's own
    ///     `config.json` is used, else `IngestOptions.default`. The choice is recorded in the report.
    public init(
        knowledgeBase: KnowledgeBase, modes: [EvalMode]? = nil,
        ingestOptions: IngestOptions? = nil
    ) {
        self.knowledgeBase = knowledgeBase
        self.requestedModes = modes
        self.ingestOverride = ingestOptions
    }

    /// Runs every mode over the answerable questions. A `.folder` dataset's documents are added first
    /// (documents already present are skipped), and a document that cannot be added fails the run.
    ///
    /// - Parameters:
    ///   - sections: what to measure; retrieval only by default.
    ///   - answerOptions: how answers are generated for the answers section. The default is repeatable: greedy decoding,
    ///     five passages, at most 300 tokens.
    ///   - verifier: cut-offs and calibration for the verifier section.
    public func run(
        dataset: Dataset, sections: EvalSections = .retrieval, answerOptions: AnswerOptions? = nil,
        verifier: VerifierOptions = VerifierOptions()
    ) async throws -> EvalReport {
        let loaded = try dataset.load()
        let ingestOptions = ingestOverride ?? Self.options(from: loaded.config)

        let ingestStart = ContinuousClock.now
        if let folder = loaded.documentsFolder {
            for try await progress in knowledgeBase.add(contentsOf: folder, options: ingestOptions) {
                if case .skipped(let reason) = progress.phase, reason != "unchanged" {
                    throw EvalError.ingestFailed(document: progress.document, reason: reason)
                }
            }
        }
        let ingestSeconds = Self.seconds(ContinuousClock.now - ingestStart)
        let stats = await knowledgeBase.stats
        var modes = requestedModes ?? [.keyword, .vector, .hybrid]
        if requestedModes == nil, stats.rerankerID != nil { modes.append(.hybridRerank) }
        if sections.contains(.retrieval), modes.contains(.hybridRerank), stats.rerankerID == nil { throw EvalError.rerankerNotConfigured }
        if sections.contains(.answers), stats.generatorID == nil { throw EvalError.missing("the answers section needs a generator in the ModelSet") }
        if sections.contains(.verifier), loaded.supportExamples.isEmpty { throw EvalError.missing("the dataset has no support.jsonl") }

        let answerable = loaded.questions.filter(\.answerable)
        var results: [ModeResult] = []
        if sections.contains(.retrieval) {
            // Load the models before timing anything.
            if let first = answerable.first {
                for mode in modes { _ = try await knowledgeBase.search(first.question, options: Self.options(mode)) }
            }
            for mode in modes {
                results.append(try await measure(mode, questions: answerable))
            }
        }

        var verifierReport: VerifierReport?
        var cutoffsForAnswers = verifier.cutoffs
        if sections.contains(.verifier) || (sections.contains(.answers) && verifier.calibrate) {
            let report = try await VerifierEvaluation.run(
                loaded.supportExamples, knowledgeBase: knowledgeBase, defaultCutoffs: verifier.cutoffs, calibrate: verifier.calibrate,
                minimumPrecision: verifier.minimumPrecision)
            if sections.contains(.verifier) { verifierReport = report }
            if let calibrated = report.calibrated { cutoffsForAnswers = calibrated.cutoffs }
        }

        var answerReport: AnswerReport?
        if sections.contains(.answers) {
            var options = answerOptions ?? AnswerOptions(
                retrieval: RetrievalOptions(limit: 5), maxAnswerTokens: 300, temperature: 0)
            if verifier.calibrate || answerOptions == nil {
                options.supportThresholds = (cutoffsForAnswers.supported, cutoffsForAnswers.weak)
            }
            answerReport = try await AnswerEvaluation.run(loaded.questions, knowledgeBase: knowledgeBase, options: options)
        }

        return EvalReport(
            dataset: DatasetSummary(
                name: loaded.name, documents: stats.documentCount, questions: loaded.questions.count,
                answerable: answerable.count, unanswerable: loaded.questions.count - answerable.count),
            environment: EnvironmentSummary(
                device: Self.deviceModel, os: ProcessInfo.processInfo.operatingSystemVersionString,
                embedder: stats.embedderID ?? "unknown", reranker: stats.rerankerID, generator: stats.generatorID, generatedAt: Date()),
            ingest: IngestSummary(
                documents: stats.documentCount, chunks: stats.chunkCount, seconds: ingestSeconds,
                chunker: Self.name(ingestOptions.chunker), targetChunkTokens: ingestOptions.targetChunkTokens,
                overlapTokens: ingestOptions.overlapTokens,
                chunksPerSecond: ingestSeconds > 0 ? Double(stats.chunkCount) / ingestSeconds : 0),
            retrieval: results, answers: answerReport, verifier: verifierReport)
    }

    private func measure(_ mode: EvalMode, questions: [EvalQuestion]) async throws -> ModeResult {
        var hits1 = 0, hits5 = 0, hits10 = 0
        var reciprocal: [Double] = [], gains: [Double] = [], latencies: [Double] = []
        var ranks: [QuestionRank] = []

        for question in questions {
            let start = ContinuousClock.now
            let found = try await knowledgeBase.search(question.question, options: Self.options(mode))
            latencies.append(Self.seconds(ContinuousClock.now - start) * 1000)

            let relevance = Relevance.flags(for: found, relevant: question.relevant)
            if Metrics.hit(relevance, atK: 1) { hits1 += 1 }
            if Metrics.hit(relevance, atK: 5) { hits5 += 1 }
            if Metrics.hit(relevance, atK: 10) { hits10 += 1 }
            reciprocal.append(Metrics.reciprocalRank(relevance))
            gains.append(Metrics.ndcg(relevance, relevantCount: question.relevant.count, k: 10))
            ranks.append(QuestionRank(id: question.id, rank: Metrics.firstRank(relevance)))
        }

        let n = Double(max(questions.count, 1))
        return ModeResult(
            mode: mode.name,
            metrics: RetrievalMetrics(
                questions: questions.count, recallAt1: Double(hits1) / n, recallAt5: Double(hits5) / n,
                recallAt10: Double(hits10) / n, mrr: Metrics.mean(reciprocal), ndcgAt10: Metrics.mean(gains),
                latencyMeanMs: Metrics.mean(latencies), latencyP50Ms: Metrics.percentile(latencies, 0.5),
                latencyP95Ms: Metrics.percentile(latencies, 0.95)),
            ranks: ranks)
    }

    private static func options(from config: DatasetConfig) -> IngestOptions {
        let defaults = IngestOptions.default
        return IngestOptions(
            chunker: config.chunker.flatMap(ChunkerKind.init(rawValue:)) ?? defaults.chunker,
            targetChunkTokens: config.chunkTokens ?? defaults.targetChunkTokens,
            overlapTokens: config.overlapTokens ?? defaults.overlapTokens)
    }

    private static func options(_ mode: EvalMode) -> RetrievalOptions {
        RetrievalOptions(limit: 10, mode: mode.retrieval, rerank: mode.rerank)
    }

    private static func name(_ kind: ChunkerKind) -> String { kind.rawValue }

    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    static var deviceModel: String {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        guard size > 0 else { return "unknown" }
        var buffer = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &buffer, &size, nil, 0)
        return String(decoding: buffer.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}
