import Foundation

/// A pre-load estimate of the memory a model needs, so loading can be refused instead of the
/// app being terminated.
public struct MemoryEstimate: Sendable, Equatable {
    /// Model weights, taken as the GGUF file size (they are memory-mapped).
    public let weights: UInt64
    /// Key/value cache for `contextTokens`; zero for encoder-only architectures.
    public let kvCache: UInt64
    /// Scratch buffers for one batch.
    public let compute: UInt64

    public var total: UInt64 { weights + kvCache + compute }

    /// Architectures that attend bidirectionally and keep no KV cache.
    static let encoderOnly: Set<String> = [
        "bert", "nomic-bert", "jina-bert-v2", "jina-bert-v3", "roberta", "xlm-roberta", "modern-bert",
        "neo-bert", "distilbert",
    ]

    /// Estimates from the file header alone, without loading weights. It is a heuristic and errs high;
    /// `LlamaModelMemoryTests` compares it with measured footprints.
    public static func estimate(modelAt path: String, contextTokens: Int) throws -> MemoryEstimate {
        estimate(header: try GGUFHeader.read(path: path), contextTokens: contextTokens)
    }

    static func estimate(header: GGUFHeader, contextTokens: Int) -> MemoryEstimate {
        let embedding = UInt64(header.embeddingLength ?? 4096)
        let tokens = UInt64(max(contextTokens, 1))

        var kv: UInt64 = 0
        if !encoderOnly.contains(header.architecture ?? "") {
            let layers = UInt64(header.blockCount ?? 32)
            let heads = UInt64(max(header.headCount ?? 32, 1))
            let kvHeads = UInt64(header.headCountKV ?? Int(heads))
            let keyLength = UInt64(header.keyLength ?? Int(embedding / heads))
            let valueLength = UInt64(header.valueLength ?? Int(embedding / heads))
            kv = layers * tokens * kvHeads * (keyLength + valueLength) * 2  // f16
        }

        let compute = UInt64(32 << 20) + tokens * embedding * 16
        return MemoryEstimate(weights: header.fileSize, kvCache: kv, compute: compute)
    }
}
