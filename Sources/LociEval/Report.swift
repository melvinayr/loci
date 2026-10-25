import Foundation

/// How `EvalReport.write` serialises a report.
public enum ReportFormat: Sendable {
    case markdown, json
}

/// What was evaluated.
public struct DatasetSummary: Codable, Sendable, Equatable {
    public var name: String
    public var documents: Int
    public var questions: Int
    public var answerable: Int
    public var unanswerable: Int
}

/// Where and with what the evaluation ran.
public struct EnvironmentSummary: Codable, Sendable, Equatable {
    public var device: String
    public var os: String
    /// The embedder ID recorded in the knowledge base.
    public var embedder: String
    /// The reranker's ID, when one was configured.
    public var reranker: String?
    /// The generator's ID, when one was configured.
    public var generator: String?
    public var generatedAt: Date
}

/// How the dataset documents were ingested.
public struct IngestSummary: Codable, Sendable, Equatable {
    public var documents: Int
    public var chunks: Int
    public var seconds: Double
    public var chunker: String
    public var targetChunkTokens: Int
    public var overlapTokens: Int
    public var chunksPerSecond: Double
}

/// The 1-based rank of the first relevant passage for one question, nil when none was retrieved.
public struct QuestionRank: Codable, Sendable, Equatable {
    public var id: String
    public var rank: Int?
}

/// Retrieval results for one mode.
public struct ModeResult: Codable, Sendable, Equatable {
    /// `keyword`, `vector`, `hybrid` or `hybrid+rerank`.
    public var mode: String
    public var metrics: RetrievalMetrics
    public var ranks: [QuestionRank]
}

/// The outcome of one evaluation run. Every number in docs and READMEs must come from a report like this.
public struct EvalReport: Codable, Sendable, Equatable {
    public var schemaVersion = 1
    public var dataset: DatasetSummary
    public var environment: EnvironmentSummary
    public var ingest: IngestSummary
    /// Empty when the retrieval section did not run.
    public var retrieval: [ModeResult]
    /// Present when the answers section ran.
    public var answers: AnswerReport?
    /// Present when the verifier section ran.
    public var verifier: VerifierReport?

    public init(
        dataset: DatasetSummary, environment: EnvironmentSummary, ingest: IngestSummary, retrieval: [ModeResult],
        answers: AnswerReport? = nil, verifier: VerifierReport? = nil
    ) {
        self.dataset = dataset
        self.environment = environment
        self.ingest = ingest
        self.retrieval = retrieval
        self.answers = answers
        self.verifier = verifier
    }

    /// The result for a mode name such as `"hybrid"`, if that mode was run.
    public func result(for mode: String) -> ModeResult? {
        retrieval.first { $0.mode == mode }
    }

    // MARK: Writing

    /// Writes the report atomically in `format`.
    public func write(to url: URL, format: ReportFormat) throws {
        let data: Data
        switch format {
        case .markdown: data = Data(markdown().utf8)
        case .json: data = try jsonData()
        }
        try data.write(to: url, options: .atomic)
    }

    /// The report as pretty-printed JSON with sorted keys, suitable for a CI baseline.
    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(self)
    }

    public static func read(from url: URL) throws -> EvalReport {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(EvalReport.self, from: Data(contentsOf: url))
    }

    /// A human-readable report.
    public func markdown() -> String {
        var lines = [
            "# Loci evaluation report",
            "",
            "- **Dataset:** \(dataset.name) — \(dataset.documents) documents, \(dataset.questions) questions "
                + "(\(dataset.answerable) answerable, \(dataset.unanswerable) unanswerable)",
            "- **Embedder:** \(environment.embedder)",
            "- **Reranker:** \(environment.reranker ?? "none")",
            "- **Generator:** \(environment.generator ?? "none")",
            "- **Device:** \(environment.device) · \(environment.os)",
            "- **Generated:** \(ISO8601DateFormatter().string(from: environment.generatedAt))",
            "- **Ingest:** \(ingest.documents) documents → \(ingest.chunks) chunks in "
                + String(format: "%.1f", ingest.seconds) + " s "
                + "(\(ingest.chunker) chunks, target \(ingest.targetChunkTokens) tokens, overlap \(ingest.overlapTokens))",
        ]
        if !retrieval.isEmpty { lines += retrievalMarkdown() }
        if let answers { lines += Self.answersMarkdown(answers) }
        if let verifier { lines += Self.verifierMarkdown(verifier) }
        lines.append("")
        return lines.joined(separator: "\n")
    }

    private static func percent(_ value: Double) -> String { String(format: "%.1f%%", value * 100) }
    private static func number(_ value: Double) -> String { String(format: "%.3f", value) }
    private static func millis(_ value: Double) -> String { String(format: "%.0f", value) }

    private func retrievalMarkdown() -> [String] {
        var lines = [
            "",
            "## Retrieval",
            "",
            "Over the \(dataset.answerable) answerable questions. Recall@k is the fraction of questions with a relevant passage in the top k.",
            "",
            "| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for result in retrieval {
            let m = result.metrics
            lines.append(
                "| \(result.mode) | \(Self.percent(m.recallAt1)) | \(Self.percent(m.recallAt5)) | \(Self.percent(m.recallAt10)) "
                    + "| \(Self.number(m.mrr)) | \(Self.number(m.ndcgAt10)) | \(Self.millis(m.latencyP50Ms)) | \(Self.millis(m.latencyP95Ms)) |")
        }
        lines += ["", "## Questions without a relevant passage in the top 5", ""]
        for result in retrieval {
            let missed = result.ranks.filter { ($0.rank ?? Int.max) > 5 }
            let detail = missed.isEmpty
                ? "none"
                : missed.map { "\($0.id) (\($0.rank.map { "rank \($0)" } ?? "not retrieved"))" }.joined(separator: ", ")
            lines.append("- **\(result.mode):** \(detail)")
        }
        return lines
    }

    private static func answersMarkdown(_ a: AnswerReport) -> [String] {
        let s = a.settings
        return [
            "",
            "## Answers",
            "",
            "Prompt v\(s.promptVersion) · \(s.retrievalLimit) passages\(s.rerank ? " reranked" : "") · temperature \(s.temperature) · "
                + "max \(s.maxAnswerTokens) tokens · verification \(s.verify ? "on" : "off") "
                + "(supported ≥ \(String(format: "%.3f", s.supportedThreshold)), weak ≥ \(String(format: "%.3f", s.weakThreshold)))",
            "",
            "### Citations",
            "",
            "| Metric | Value |",
            "|---|---|",
            "| Citation precision | \(percent(a.citations.precision)) (of \(a.citations.count) citations) |",
            "| Citation recall | \(percent(a.citations.recall)) (of \(a.answerable) answerable questions) |",
            "| Structural validity | \(percent(a.citations.structuralValidity)) (of \(a.citations.checked) citations; must be 100%) |",
            "",
            "### Faithfulness",
            "",
            "| Metric | Value |",
            "|---|---|",
            "| Supported sentences | \(percent(a.faithfulness.supportedFraction)) (of \(a.faithfulness.sentences)) |",
            "| Weak sentences | \(percent(a.faithfulness.weakFraction)) |",
            "| Unsupported rate | \(percent(a.faithfulness.unsupportedRate)) |",
            "",
            "### Refusals and answer match",
            "",
            "| Metric | Value |",
            "|---|---|",
            "| Refusal accuracy | \(percent(a.refusal.accuracy)) (of \(a.refusal.unanswerable) unanswerable questions) |",
            "| False refusal rate | \(percent(a.refusal.falseRefusalRate)) (of \(a.refusal.answerable) answerable questions) |",
            "| Answer contains the reference | \(percent(a.answerMatch.fraction)) (of \(a.answerMatch.withReference); a simple containment check, so a correct paraphrase counts as a miss) |",
            "",
            "### Performance",
            "",
            "| Stage | p50 | p95 |",
            "|---|---|---|",
            "| Retrieval | \(millis(a.performance.retrievalP50Ms)) ms | |",
            "| Rerank | \(millis(a.performance.rerankP50Ms)) ms | |",
            "| Time to first token | \(millis(a.performance.timeToFirstTokenP50Ms)) ms | \(millis(a.performance.timeToFirstTokenP95Ms)) ms |",
            "| Verification | \(millis(a.performance.verificationP50Ms)) ms | |",
            "| Whole answer | \(millis(a.performance.totalP50Ms)) ms | \(millis(a.performance.totalP95Ms)) ms |",
            "",
            String(format: "Generation %.1f tokens/s · peak memory %.0f MB (process so far) · index %.0f KB",
                   a.performance.tokensPerSecondMean, Double(a.performance.peakMemoryBytes) / 1_048_576, Double(a.performance.indexBytes) / 1024),
        ]
    }

    private static func verifierMarkdown(_ v: VerifierReport) -> [String] {
        func row(_ name: String, _ m: VerifierMetrics) -> String {
            "| \(name) | \(m.examples) | \(percent(m.accuracy)) | \(percent(m.unsupportedPrecision)) | \(percent(m.unsupportedRecall)) | "
                + "\(number(m.unsupportedF1)) | \(number(m.supportedF1)) |"
        }
        func cutoffs(_ c: SupportCutoffs) -> String {
            "supported ≥ \(String(format: "%.3f", c.supported)), weak ≥ \(String(format: "%.3f", c.weak))"
        }
        let order = SupportExample.Label.allCases
        var lines = [
            "",
            "## Verification against hand labels",
            "",
            "Scorer: \(v.scorer). \(v.examples) labelled (claim, passage) pairs. Mean score by human label: "
                + order.map { "\($0.rawValue) \(String(format: "%.3f", v.meanScore[$0.rawValue] ?? 0))" }.joined(separator: ", ")
                + ". Probability a supported example outscores an unsupported one: \(number(v.aucSupportedVsUnsupported)); "
                + "outscores a weak one: \(number(v.aucSupportedVsWeak)) (0.5 is chance).",
            "",
            "| Cut-offs | Examples | Accuracy | Unsupported precision | Unsupported recall | Unsupported F1 | Supported F1 |",
            "|---|---|---|---|---|---|---|",
            row("default (\(cutoffs(v.defaultCutoffs))) — calibration half", v.atDefault.calibration),
            row("default — test half", v.atDefault.test),
        ]
        if let calibrated = v.calibrated {
            lines += [
                row("**calibrated (\(cutoffs(calibrated.cutoffs))) — calibration half (fitted here)**", calibrated.calibration),
                row("**calibrated — test half (not used for fitting)**", calibrated.test),
            ]
        }
        if v.calibrated != nil {
            lines += ["", "Calibrated `weak` flags as many unsupported claims as it can while at least \(percent(v.minimumPrecision)) of those flagged really are unsupported; "
                + "when the scorer cannot separate weak from supported claims the two cut-offs are equal."]
        }
        let confusion = (v.calibrated?.test ?? v.atDefault.test).confusion
        lines += ["", "Test-half confusion (rows: human label, columns: predicted; order supported, weak, unsupported):", ""]
        for (index, counts) in confusion.enumerated() {
            lines.append("- \(order[index].rawValue): \(counts.map(String.init).joined(separator: " / "))")
        }
        return lines
    }

    // MARK: Regression

    /// Differences that break the CI policy against `baseline`: Recall@5 down by more than `margin` (absolute,
    /// 0.02 = two percentage points) for any mode in both reports, citation precision down by more than `margin` when both
    /// reports have answers, and structural validity below 100% in this report.
    public func regressions(against baseline: EvalReport, margin: Double = 0.02) -> [String] {
        var found: [String] = []
        for old in baseline.retrieval {
            guard let new = result(for: old.mode) else { continue }
            let drop = old.metrics.recallAt5 - new.metrics.recallAt5
            if drop > margin + 1e-9 {
                found.append(String(
                    format: "%@ Recall@5 fell from %.1f%% to %.1f%% (allowed drop %.1f points)",
                    old.mode, old.metrics.recallAt5 * 100, new.metrics.recallAt5 * 100, margin * 100))
            }
        }
        if let old = baseline.answers, let new = answers {
            let drop = old.citations.precision - new.citations.precision
            if drop > margin + 1e-9 {
                found.append(String(
                    format: "Citation precision fell from %.1f%% to %.1f%% (allowed drop %.1f points)",
                    old.citations.precision * 100, new.citations.precision * 100, margin * 100))
            }
        }
        if let answers, answers.citations.structuralValidity < 1 {
            found.append(String(
                format: "Structural validity is %.1f%%; it must be 100%%", answers.citations.structuralValidity * 100))
        }
        return found
    }
}
