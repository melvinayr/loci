import Foundation
import Loci

/// What to measure and how much of it.
public struct BenchmarkOptions: Sendable {
    /// Passages in the synthetic corpus used for ingest and search timings. 50,000 is the target size for vector search.
    public var chunkCount: Int
    /// Queries timed per search mode.
    public var queries: Int
    /// Queries timed for the (much slower) reranked mode.
    public var rerankQueries: Int
    /// Dataset whose questions time full answers; nil skips answers.
    public var answerDataset: URL?

    public init(chunkCount: Int = 10_000, queries: Int = 100, rerankQueries: Int = 10, answerDataset: URL? = nil) {
        self.chunkCount = chunkCount
        self.queries = queries
        self.rerankQueries = rerankQueries
        self.answerDataset = answerDataset
    }
}

public struct BenchmarkReport: Codable, Sendable, Equatable {
    public struct Timing: Codable, Sendable, Equatable {
        public var name: String
        public var samples: Int
        public var meanMs: Double
        public var p50Ms: Double
        public var p95Ms: Double
        public var maxMs: Double
    }

    public struct Ingest: Codable, Sendable, Equatable {
        public var chunks: Int
        public var seconds: Double
        public var chunksPerSecond: Double
        public var indexBytes: UInt64
    }

    public struct MemoryPoint: Codable, Sendable, Equatable {
        public var stage: String
        public var footprintBytes: UInt64
    }

    public var environment: EnvironmentSummary
    public var options: Options
    public struct Options: Codable, Sendable, Equatable {
        public var chunkCount: Int
        public var queries: Int
        public var rerankQueries: Int
    }
    /// Time for the first use of each model after it was unloaded: loading plus one operation.
    public var coldStarts: [Timing]
    public var ingest: Ingest
    /// Search latency over the synthetic corpus. Vector and hybrid include embedding the query.
    public var search: [Timing]
    /// Present when an answer dataset was given.
    public var answers: AnswerReport.Performance?
    public var memory: [MemoryPoint]
    /// Peak footprint of the process over the whole run.
    public var peakFootprintBytes: UInt64
}

/// Times the pipeline on this machine. Every number it produces comes from a run; none is written by hand.
public struct Benchmark: Sendable {
    private let makeKnowledgeBase: @Sendable () async throws -> KnowledgeBase

    /// - Parameter makeKnowledgeBase: creates an empty, scratch knowledge base with the models to benchmark. Called more than once.
    public init(makeKnowledgeBase: @escaping @Sendable () async throws -> KnowledgeBase) {
        self.makeKnowledgeBase = makeKnowledgeBase
    }

    public func run(_ options: BenchmarkOptions = BenchmarkOptions()) async throws -> BenchmarkReport {
        var memory: [BenchmarkReport.MemoryPoint] = []
        func note(_ stage: String) { memory.append(.init(stage: stage, footprintBytes: MemoryUsage.currentBytes)) }
        note("start")

        let kb = try await makeKnowledgeBase()
        let corpus = SyntheticCorpus(seed: 42)
        let text = corpus.document(chunks: options.chunkCount)

        // Ingest: one document of many passages, so the time is parsing, chunking, embedding and storing, not 50,000 commits.
        let ingestStart = ContinuousClock.now
        try await kb.add(.text(text, title: "synthetic"), options: IngestOptions(targetChunkTokens: 64, overlapTokens: 0))
        let ingestSeconds = Self.seconds(ContinuousClock.now - ingestStart)
        let stats = await kb.stats
        note("after ingest (embedder resident)")

        let queries = corpus.queries(options.queries)
        var search: [BenchmarkReport.Timing] = []

        // The first vector search also loads every vector into memory.
        let firstVector = try await Self.time { _ = try await kb.search(queries[0], options: RetrievalOptions(mode: .vector, rerank: false)) }
        var cold: [BenchmarkReport.Timing] = [Self.timing("first vector search after ingest (loads the vector snapshot)", [firstVector])]

        for (name, mode) in [("keyword", RetrievalMode.keyword), ("vector", .vector), ("hybrid", .hybrid)] {
            let times = try await Self.times(queries) { query in
                _ = try await kb.search(query, options: RetrievalOptions(limit: 5, mode: mode, rerank: false))
            }
            search.append(Self.timing("\(name) search", times))
        }
        note("after search")

        if stats.rerankerID != nil, options.rerankQueries > 0 {
            await kb.unloadModels()
            let coldRerank = try await Self.time { _ = try await kb.search(queries[0], options: RetrievalOptions(limit: 5, mode: .hybrid, rerank: true)) }
            cold.append(Self.timing("first reranked search after unloading models (loads embedder and reranker)", [coldRerank]))
            let times = try await Self.times(Array(queries.prefix(options.rerankQueries))) { query in
                _ = try await kb.search(query, options: RetrievalOptions(limit: 5, mode: .hybrid, rerank: true))
            }
            search.append(Self.timing("hybrid + rerank search (20 candidates reranked)", times))
            note("after reranked search")
        }

        var answerPerformance: AnswerReport.Performance?
        if let dataset = options.answerDataset, stats.generatorID != nil {
            let answerKB = try await makeKnowledgeBase()
            let report = try await Evaluator(knowledgeBase: answerKB, modes: [])
                .run(dataset: .folder(dataset), sections: .answers)
            answerPerformance = report.answers?.performance
            note("after answers")
        }

        return BenchmarkReport(
            environment: EnvironmentSummary(
                device: Evaluator.deviceModel, os: ProcessInfo.processInfo.operatingSystemVersionString,
                embedder: stats.embedderID ?? "unknown", reranker: stats.rerankerID, generator: stats.generatorID, generatedAt: Date()),
            options: .init(chunkCount: options.chunkCount, queries: options.queries, rerankQueries: options.rerankQueries),
            coldStarts: cold,
            ingest: .init(
                chunks: stats.chunkCount, seconds: ingestSeconds,
                chunksPerSecond: ingestSeconds > 0 ? Double(stats.chunkCount) / ingestSeconds : 0, indexBytes: stats.databaseBytes),
            search: search, answers: answerPerformance, memory: memory, peakFootprintBytes: MemoryUsage.peakBytes)
    }

    // MARK: Timing helpers

    private static func time(_ body: () async throws -> Void) async throws -> Double {
        let start = ContinuousClock.now
        try await body()
        return seconds(ContinuousClock.now - start) * 1000
    }

    private static func times(_ inputs: [String], _ body: (String) async throws -> Void) async throws -> [Double] {
        var result: [Double] = []
        for input in inputs { result.append(try await time { try await body(input) }) }
        return result
    }

    private static func timing(_ name: String, _ values: [Double]) -> BenchmarkReport.Timing {
        .init(
            name: name, samples: values.count, meanMs: Metrics.mean(values), p50Ms: Metrics.percentile(values, 0.5),
            p95Ms: Metrics.percentile(values, 0.95), maxMs: values.max() ?? 0)
    }

    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    // MARK: Writing

    public static func markdown(_ report: BenchmarkReport) -> String {
        func ms(_ value: Double) -> String { value >= 100 ? String(format: "%.0f", value) : String(format: "%.1f", value) }
        func mb(_ bytes: UInt64) -> String { String(format: "%.0f MB", Double(bytes) / 1_048_576) }
        let e = report.environment
        var lines = [
            "# Benchmarks",
            "",
            "Generated by `loci-cli bench`; every number below comes from that run on the machine described here, not from anywhere else. "
                + "Results on other devices (in particular iPhones) will differ and have not been measured.",
            "",
            "- **Device:** \(e.device) · \(e.os)",
            "- **Embedder:** \(e.embedder)",
            "- **Reranker:** \(e.reranker ?? "none")",
            "- **Generator:** \(e.generator ?? "none")",
            "- **Generated:** \(ISO8601DateFormatter().string(from: e.generatedAt))",
            "- **Corpus:** one synthetic document of \(report.ingest.chunks) passages (about 50 tokens each) of random sentences drawn from a fixed vocabulary, seeded so runs are comparable",
            "",
            "## Ingest",
            "",
            "| Passages | Time | Throughput | Index on disk |",
            "|---|---|---|---|",
            "| \(report.ingest.chunks) | \(String(format: "%.1f", report.ingest.seconds)) s | \(String(format: "%.0f", report.ingest.chunksPerSecond)) passages/s | \(mb(report.ingest.indexBytes)) |",
            "",
            "Ingest covers parsing, chunking with the embedder's tokenizer, embedding and one SQLite transaction.",
            "",
            "## Search latency over \(report.ingest.chunks) passages",
            "",
            "| Search | Samples | Mean ms | p50 ms | p95 ms | Max ms |",
            "|---|---|---|---|---|---|",
        ]
        for t in report.search { lines.append("| \(t.name) | \(t.samples) | \(ms(t.meanMs)) | \(ms(t.p50Ms)) | \(ms(t.p95Ms)) | \(ms(t.maxMs)) |") }
        lines += [
            "",
            "Vector and hybrid search include embedding the query; keyword search involves no model. The project target is under 50 ms for vector search over 50,000 passages on a recent iPhone; "
                + "the vector rows are the corresponding measurement on this machine, with the query embedding included.",
            "",
            "## Cold starts",
            "",
            "| First use | Time |",
            "|---|---|",
        ]
        for t in report.coldStarts { lines.append("| \(t.name) | \(ms(t.meanMs)) ms |") }
        if let a = report.answers {
            lines += [
                "",
                "## Answers",
                "",
                "| Stage | p50 | p95 |",
                "|---|---|---|",
                "| Retrieval | \(ms(a.retrievalP50Ms)) ms | |",
                "| Rerank | \(ms(a.rerankP50Ms)) ms | |",
                "| Time to first token | \(ms(a.timeToFirstTokenP50Ms)) ms | \(ms(a.timeToFirstTokenP95Ms)) ms |",
                "| Verification | \(ms(a.verificationP50Ms)) ms | |",
                "| Whole answer | \(ms(a.totalP50Ms)) ms | \(ms(a.totalP95Ms)) ms |",
                "",
                String(format: "Generation %.1f tokens/s on average.", a.tokensPerSecondMean),
            ]
        }
        lines += ["", "## Memory", "", "Process footprint as the run progressed (macOS `phys_footprint`).", "", "| Stage | Footprint |", "|---|---|"]
        for point in report.memory { lines.append("| \(point.stage) | \(mb(point.footprintBytes)) |") }
        lines += ["", "Peak footprint over the run: \(mb(report.peakFootprintBytes)).", ""]
        return lines.joined(separator: "\n")
    }
}

/// Deterministic synthetic text: random sentences from a fixed vocabulary.
struct SyntheticCorpus {
    private var state: UInt64
    private static let vocabulary: [String] = """
        the of and to in a is that for it as was with be by on not he this are or his from at which but have an had they you were their one all we can her has \
        there been if more when will would who so no she other its may these two time first only new some could into than out about then them like over such \
        also back after use how our work well way even want because any those very through between before under around lighthouse starter inverter trial coral \
        network guild transit participants engineer survey keeper bread panel route adverse station reef apprentice firmware warranty battery charge tower \
        results methods outcome dough flour water oven bridge ledger hall stone granite lantern signal harbour council budget fare lane metro bus station \
        sample cover sites divers fish biomass bleaching channel router mesh node extender passphrase masters wardens cloth ell fine seal charter
        """.split(separator: " ").map(String.init)

    init(seed: UInt64) { state = seed &* 0x9E3779B97F4A7C15 &+ 1 }

    private mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    private mutating func word() -> String { Self.vocabulary[Int(next() % UInt64(Self.vocabulary.count))] }

    /// About 12 words, capitalised and ended with a full stop.
    private mutating func sentence() -> String {
        var words = (0..<12).map { _ in word() }
        words[0] = words[0].prefix(1).uppercased() + words[0].dropFirst()
        return words.joined(separator: " ") + "."
    }

    /// `chunks` paragraphs of three sentences each (about 50 tokens), which chunk to about one passage per paragraph at a 64-token target.
    func document(chunks: Int) -> String {
        var copy = self
        return (0..<max(chunks, 1)).map { index in
            "Passage \(index). " + (0..<3).map { _ in copy.sentence() }.joined(separator: " ")
        }.joined(separator: "\n\n")
    }

    func queries(_ count: Int) -> [String] {
        var copy = self
        copy.state ^= 0xA5A5A5A5
        return (0..<max(count, 1)).map { _ in (0..<6).map { _ in copy.word() }.joined(separator: " ") }
    }
}
