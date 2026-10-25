/// Reciprocal Rank Fusion: combines rankings without normalising their scores.
///
/// Each item scores `Σ 1 / (k + rank)` over the rankings it appears in (rank is 1-based).
public enum RRF {
    public static let defaultK = 60

    /// Fused ids, best first. Ties keep the order in which items first appeared, so results are
    /// deterministic.
    public static func fuse<ID: Hashable>(_ rankings: [[ID]], k: Int = defaultK) -> [(id: ID, score: Double)] {
        var scores: [ID: Double] = [:]
        var firstSeen: [ID: Int] = [:]
        for ranking in rankings {
            var seenInRanking = Set<ID>()
            for (offset, id) in ranking.enumerated() where seenInRanking.insert(id).inserted {
                scores[id, default: 0] += 1 / Double(k + offset + 1)
                if firstSeen[id] == nil { firstSeen[id] = firstSeen.count }
            }
        }
        return scores
            .map { (id: $0.key, score: $0.value) }
            .sorted { lhs, rhs in
                lhs.score != rhs.score ? lhs.score > rhs.score : (firstSeen[lhs.id] ?? 0) < (firstSeen[rhs.id] ?? 0)
            }
    }
}
