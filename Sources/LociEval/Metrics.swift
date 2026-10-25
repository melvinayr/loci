import Foundation

/// Retrieval metrics for one mode, over the answerable questions.
public struct RetrievalMetrics: Codable, Sendable, Equatable {
    public var questions: Int
    public var recallAt1: Double
    /// Fraction of questions with a relevant passage among the first 5 results (likewise `recallAt1` and `recallAt10`).
    public var recallAt5: Double
    public var recallAt10: Double
    /// Mean reciprocal rank of the first relevant passage (0 when none is retrieved).
    public var mrr: Double
    /// Normalised discounted cumulative gain over the first 10 results, with binary relevance.
    public var ndcgAt10: Double
    /// Retrieval latency per query, including query embedding.
    public var latencyMeanMs: Double
    /// Median retrieval time in milliseconds (`latencyP95Ms` is the 95th percentile).
    public var latencyP50Ms: Double
    public var latencyP95Ms: Double
}

/// The pure arithmetic behind the metrics.
enum Metrics {
    /// `relevance[i]` is true when the i-th result matched a not-yet-matched relevant passage.
    static func hit(_ relevance: [Bool], atK k: Int) -> Bool {
        relevance.prefix(k).contains(true)
    }

    /// 1-based rank of the first relevant result, or nil.
    static func firstRank(_ relevance: [Bool]) -> Int? {
        relevance.firstIndex(of: true).map { $0 + 1 }
    }

    static func reciprocalRank(_ relevance: [Bool]) -> Double {
        firstRank(relevance).map { 1 / Double($0) } ?? 0
    }

    /// Binary-gain nDCG: each relevant passage can be credited once.
    static func ndcg(_ relevance: [Bool], relevantCount: Int, k: Int) -> Double {
        guard relevantCount > 0 else { return 0 }
        func discount(_ position: Int) -> Double { 1 / log2(Double(position) + 2) }  // position is 0-based
        let dcg = relevance.prefix(k).enumerated().reduce(0.0) { $0 + ($1.element ? discount($1.offset) : 0) }
        let ideal = (0..<min(relevantCount, k)).reduce(0.0) { $0 + discount($1) }
        return dcg / ideal
    }

    /// Nearest-rank percentile of `values`, with `p` in 0...1.
    static func percentile(_ values: [Double], _ p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let rank = Int((p * Double(sorted.count)).rounded(.up))
        return sorted[min(max(rank - 1, 0), sorted.count - 1)]
    }

    static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
    }
}
