import LociStore

/// A chunk with the scores it earned, in priority order.
struct Candidate {
    var chunk: StoredChunk
    var bm25: Float?
    var vector: Float?
    var fused: Float?
    var rerank: Float?
}

/// Selects up to `limit` passages from prioritised candidates, merging chunks of the same
/// document that overlap or touch (and, for PDFs, share a page) into one passage.
enum PassageMerger {
    /// Chunks closer than this many scalars apart count as adjacent: the gap is at most one
    /// block separator ("\n\n") or a single space between sentences.
    static let adjacencyGap = 2

    private struct Group {
        var documentID: String
        var page: Int?
        var range: Range<Int>
        var chunks: [StoredChunk]
        var best: Candidate
        var priority: Int
    }

    static func merge(
        _ candidates: [Candidate], limit: Int,
        documentText: (String) async throws -> String?
    ) async throws -> [Passage] {
        guard limit > 0 else { return [] }
        var groups: [Group] = []

        for (priority, candidate) in candidates.enumerated() {
            let chunk = candidate.chunk
            let range = chunk.start..<chunk.end
            let touching = groups.indices.filter { index in
                let group = groups[index]
                return group.documentID == chunk.documentID && group.page == chunk.page
                    && range.lowerBound <= group.range.upperBound + adjacencyGap
                    && group.range.lowerBound <= range.upperBound + adjacencyGap
            }
            if touching.isEmpty {
                if groups.count < limit {
                    groups.append(Group(
                        documentID: chunk.documentID, page: chunk.page, range: range, chunks: [chunk],
                        best: candidate, priority: priority))
                }
                continue
            }
            // Fold the candidate and every group it bridges into the first of them.
            var merged = groups[touching[0]]
            merged.chunks.append(chunk)
            merged.range = min(merged.range.lowerBound, range.lowerBound)..<max(merged.range.upperBound, range.upperBound)
            for index in touching.dropFirst() {
                let other = groups[index]
                merged.chunks += other.chunks
                merged.range = min(merged.range.lowerBound, other.range.lowerBound)..<max(merged.range.upperBound, other.range.upperBound)
                if other.priority < merged.priority {
                    merged.best = other.best
                    merged.priority = other.priority
                }
            }
            groups[touching[0]] = merged
            for index in touching.dropFirst().reversed() { groups.remove(at: index) }
        }

        groups.sort { $0.priority < $1.priority }
        var passages: [Passage] = []
        for group in groups {
            let first = group.chunks.min { $0.start < $1.start } ?? group.best.chunk
            var text = group.best.chunk.text
            if group.chunks.count > 1 {
                guard let full = try await documentText(group.documentID) else { continue }
                text = slice(full, group.range)
            }
            passages.append(Passage(
                chunkID: group.best.chunk.id, documentID: group.documentID, page: group.page, range: group.range,
                text: text, headingPath: first.headingPath, bm25: group.best.bm25, vector: group.best.vector,
                fused: group.best.fused, rerank: group.best.rerank))
        }
        return passages
    }

    private static func slice(_ text: String, _ range: Range<Int>) -> String {
        let scalars = text.unicodeScalars
        guard range.lowerBound >= 0, range.upperBound <= scalars.count else { return "" }
        let start = scalars.index(scalars.startIndex, offsetBy: range.lowerBound)
        let end = scalars.index(start, offsetBy: range.count)
        return String(scalars[start..<end])
    }
}
