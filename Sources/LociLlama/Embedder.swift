import Foundation
import llama

/// Identifies an embedding model and the shape of its vectors. Vectors from different
/// identities are not comparable, so the store records the identity it was built with.
public struct EmbedderIdentity: Sendable, Hashable, Codable {
    public let id: String
    public let dimension: Int

    public init(id: String, dimension: Int) {
        self.id = id
        self.dimension = dimension
    }
}

public enum EmbeddingRole: Sendable {
    case query
    case document
}

/// Turns text into L2-normalised vectors. `Embedder` is the llama.cpp implementation;
/// tests substitute deterministic fakes.
public protocol TextEmbedder: Sendable {
    var identity: EmbedderIdentity { get }
    /// One unit-length vector per text, in order.
    func embed(_ texts: [String], role: EmbeddingRole) async throws -> [[Float]]
    /// Counts tokens in a run of scalars; safe to call from any thread.
    func countTokens(_ scalars: ArraySlice<Unicode.Scalar>) -> Int
}

/// Scores how well passages answer a query with a cross-encoder. `Reranker` is the llama.cpp
/// implementation; tests substitute deterministic fakes.
public protocol TextReranker: Sendable {
    /// Identifies the model; score scales differ between rerankers, so thresholds are per ID.
    var id: String { get }
    /// One relevance score per passage, in order. Higher means more relevant; the scale is the
    /// model's raw logit, not a probability.
    ///
    /// `instruction` tells an instruction-aware reranker what kind of match to look for (relevance, support, …). Models that
    /// take no instruction ignore it; nil keeps the model's own default.
    func score(query: String, passages: [String], instruction: String?) async throws -> [Float]
}

extension TextReranker {
    /// Scores with the model's default instruction.
    public func score(query: String, passages: [String]) async throws -> [Float] {
        try await score(query: query, passages: passages, instruction: nil)
    }
}

/// Query/document prefixes that embedding models were trained with.
struct EmbeddingPrompts: Equatable {
    var query = ""
    var document = ""

    static func forModel(name: String, architecture: String) -> EmbeddingPrompts {
        let key = (name + " " + architecture).lowercased()
        if key.contains("embeddinggemma") || key.contains("gemma-embedding") {
            return EmbeddingPrompts(query: "task: search result | query: ", document: "title: none | text: ")
        }
        if key.contains("bge") {
            return EmbeddingPrompts(query: "Represent this sentence for searching relevant passages: ")
        }
        if key.contains("nomic") {
            return EmbeddingPrompts(query: "search_query: ", document: "search_document: ")
        }
        if key.contains("e5") {
            return EmbeddingPrompts(query: "query: ", document: "passage: ")
        }
        if key.contains("qwen3") && key.contains("embed") {
            return EmbeddingPrompts(
                query: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ")
        }
        return EmbeddingPrompts()
    }
}

/// A GGUF embedding model on llama.cpp.
public final class Embedder: TextEmbedder, @unchecked Sendable {
    // Safety of @unchecked Sendable: `context` is only used inside `LlamaExecutor.run`, which
    // executes on one serial queue, so it is never touched concurrently. Everything else is a
    // `let` of an immutable value or the read-only `LlamaModel`.
    public let identity: EmbedderIdentity
    private let model: LlamaModel
    private let context: LlamaContext
    private let prompts: EmbeddingPrompts
    private let maxSequenceTokens: Int

    private init(model: LlamaModel, context: LlamaContext, identity: EmbedderIdentity, maxSequenceTokens: Int) {
        self.model = model
        self.context = context
        self.identity = identity
        self.prompts = .forModel(name: model.name, architecture: model.architecture)
        self.maxSequenceTokens = maxSequenceTokens
    }

    /// Loads the model and creates its context on the llama queue.
    ///
    /// Throws `LlamaError.roleMismatch` if the model has no pooling (it is not an embedding model).
    public static func load(path: String, gpuLayers: Int32) async throws -> Embedder {
        try await LlamaExecutor.run { _ in
            try GGUFHeader.read(path: path).requireEmbedderRole()
            let model = try LlamaModel(path: path, gpuLayers: gpuLayers)
            let batchTokens = max(8, min(model.trainingContext, 2048))
            let context = try LlamaContext(
                model: model, contextTokens: batchTokens, batchTokens: batchTokens, maxSequences: 8, embeddings: true,
                cpuOnly: gpuLayers == 0)
            guard context.poolingType != LLAMA_POOLING_TYPE_NONE,
                  context.poolingType != LLAMA_POOLING_TYPE_UNSPECIFIED,
                  context.poolingType != LLAMA_POOLING_TYPE_RANK else {
                throw LlamaError.roleMismatch(expected: "embedding")
            }
            let identity = try Self.identity(header: try GGUFHeader.read(path: path), dimension: model.embeddingDimension)
            return Embedder(
                model: model, context: context, identity: identity,
                maxSequenceTokens: min(batchTokens, 1024))
        }
    }

    /// Reads the identity from the file header without loading weights. Its `id` equals the
    /// loaded model's, so it can be compared with the identity a store was built with.
    public static func identity(ofModelAt path: String) async throws -> EmbedderIdentity {
        try await LlamaExecutor.run { _ in
            let header = try GGUFHeader.read(path: path)
            try header.requireEmbedderRole()
            return try Self.identity(header: header, dimension: header.embeddingLength)
        }
    }

    /// The id is built only from the file header (never from loaded state) so the cheap and the
    /// loaded identity always agree. `dimension` is the actual output size when the model is loaded.
    private static func identity(header: GGUFHeader, dimension: Int?) throws -> EmbedderIdentity {
        guard let length = header.embeddingLength, length > 0, let dimension, dimension > 0 else {
            throw LlamaError.roleMismatch(expected: "embedding")
        }
        let name = header.name?.isEmpty == false ? header.name ?? "unnamed" : "unnamed"
        return EmbedderIdentity(id: "\(name)|\(length)|\(header.fileSize)", dimension: dimension)
    }

    /// Estimated memory for loading the embedder at `path`, from its header alone.
    public static func estimatedMemory(modelAt path: String) throws -> UInt64 {
        let header = try GGUFHeader.read(path: path)
        try header.requireEmbedderRole()  // before memory is considered, so the error names the real problem
        return MemoryEstimate.estimate(header: header, contextTokens: 2048).total
    }

    public func countTokens(_ scalars: ArraySlice<Unicode.Scalar>) -> Int {
        model.tokenCount(scalars)
    }

    public func embed(_ texts: [String], role: EmbeddingRole) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let prefix = role == .query ? prompts.query : prompts.document
        return try await LlamaExecutor.run { isCancelled in
            let sequences = texts.map { self.tokens(for: prefix + $0) }
            var vectors: [[Float]] = []
            vectors.reserveCapacity(texts.count)

            var start = 0
            while start < sequences.count {
                if isCancelled() { throw CancellationError() }
                var end = start
                var tokens = 0
                while end < sequences.count, end - start < self.context.maxSequences,
                      tokens + sequences[end].count <= self.context.batchCapacity {
                    tokens += sequences[end].count
                    end += 1
                }
                // A sequence is never longer than the batch, so each pass takes at least one.
                let batch = Array(sequences[start..<end])
                try self.context.process(batch)
                for index in batch.indices {
                    var vector = try self.context.pooledEmbedding(sequence: index, dimension: self.identity.dimension)
                    Self.normalize(&vector)
                    vectors.append(vector)
                }
                start = end
            }
            return vectors
        }
    }

    /// Tokens for one input, truncated to the model's limit while keeping the closing special token.
    private func tokens(for text: String) -> [llama_token] {
        var tokens = model.tokenize(text, addSpecial: true)
        if tokens.isEmpty { tokens = model.tokenize(" ", addSpecial: true) }
        if tokens.count > maxSequenceTokens, let last = tokens.last {
            tokens = Array(tokens.prefix(maxSequenceTokens - 1)) + [last]
        }
        return tokens
    }

    static func normalize(_ vector: inout [Float]) {
        var sum: Float = 0
        for value in vector { sum += value * value }
        let norm = sum.squareRoot()
        guard norm > 0, norm.isFinite else { return }
        for index in vector.indices { vector[index] /= norm }
    }
}
