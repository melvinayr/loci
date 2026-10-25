import Accelerate

/// An in-memory copy of every stored vector as half precision, searched by brute force.
///
/// Rows are converted to Float32 in batches and scored against the query with one
/// matrix-vector product per batch, so memory stays at two bytes per component.
struct VectorSnapshot: Sendable {
    var dimension = 0
    var chunkIDs: [Int64] = []
    /// Index into `documentIDs` for each row.
    var documentOrdinals: [Int32] = []
    var documentIDs: [String] = []
    /// Row-major `chunkIDs.count × dimension` half-precision components.
    var components: [UInt16] = []

    var count: Int { chunkIDs.count }

    private static let batchRows = 4096

    /// The `limit` best rows by dot product (cosine, since vectors are unit length).
    func search(_ query: [Float], limit: Int, allowedDocuments: Set<String>?) -> [VectorHit] {
        guard limit > 0, count > 0, query.count == dimension, dimension > 0 else { return [] }

        var allowed: [Bool]?
        if let allowedDocuments {
            allowed = documentIDs.map { allowedDocuments.contains($0) }
        }

        var best: [VectorHit] = []  // sorted by score, descending, at most `limit` long
        var scratch = [Float](repeating: 0, count: Self.batchRows * dimension)
        var scores = [Float](repeating: 0, count: Self.batchRows)

        components.withUnsafeBufferPointer { all in
            guard let base = all.baseAddress else { return }
            var start = 0
            while start < count {
                let rows = min(Self.batchRows, count - start)
                scratch.withUnsafeMutableBufferPointer { floats in
                    guard let floatBase = floats.baseAddress else { return }
                    Float16Codec.decode(base + start * dimension, count: rows * dimension, into: floatBase)
                    // (rows × dimension) · (dimension × 1)
                    vDSP_mmul(floatBase, 1, query, 1, &scores, 1, vDSP_Length(rows), 1, vDSP_Length(dimension))
                }
                for row in 0..<rows {
                    let index = start + row
                    if let allowed, !allowed[Int(documentOrdinals[index])] { continue }
                    let score = scores[row]
                    guard score.isFinite else { continue }
                    if best.count == limit, let worst = best.last, score <= worst.score { continue }
                    let hit = VectorHit(chunkID: chunkIDs[index], score: score)
                    let position = best.firstIndex { $0.score < score } ?? best.count
                    best.insert(hit, at: position)
                    if best.count > limit { best.removeLast() }
                }
                start += rows
            }
        }
        return best
    }
}

/// Holds the snapshot and rebuilds it lazily after any write.
actor VectorCache {
    private var snapshot = VectorSnapshot()
    private var isValid = false
    private var generation = 0

    func invalidate() {
        generation += 1
        isValid = false
    }

    /// Searches, loading the snapshot first if a write has made it stale. A load that overlaps
    /// another write is discarded and repeated, so results never come from a stale snapshot.
    func search(
        _ query: [Float], limit: Int, allowedDocuments: Set<String>?,
        load: @Sendable () async throws -> VectorSnapshot
    ) async throws -> [VectorHit] {
        while !isValid {
            let startedAt = generation
            let loaded = try await load()
            if startedAt == generation {
                snapshot = loaded
                isValid = true
            }
        }
        return snapshot.search(query, limit: limit, allowedDocuments: allowedDocuments)
    }
}
