import Foundation
import llama

/// Owns one `llama_context *` and its batch, freeing both in `deinit`.
///
/// Not thread-safe: instances are only touched from `LlamaExecutor.queue`. Holds its
/// `LlamaModel` strongly so the model always outlives the context.
final class LlamaContext {
    let model: LlamaModel
    let pointer: OpaquePointer
    /// Maximum tokens per `llama_decode` / `llama_encode` call.
    let batchCapacity: Int
    let maxSequences: Int
    private var batch: llama_batch

    init(
        model: LlamaModel, contextTokens: Int, batchTokens: Int, maxSequences: Int,
        embeddings: Bool, pooling: llama_pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED, cpuOnly: Bool = false
    ) throws {
        var params = llama_context_default_params()
        params.n_ctx = UInt32(contextTokens)
        params.n_batch = UInt32(batchTokens)
        // Non-causal embedding models must process a whole sequence in one micro-batch.
        params.n_ubatch = UInt32(batchTokens)
        params.n_seq_max = UInt32(maxSequences)
        params.embeddings = embeddings
        // Share one KV buffer across sequences; otherwise each of the `maxSequences` gets only a slice of n_ctx.
        params.kv_unified = true
        if cpuOnly {
            // No GPU layers means CPU inference: do not let large batches be shipped to the GPU either.
            params.op_offload = false
            params.offload_kqv = false
        }
        params.pooling_type = pooling  // UNSPECIFIED takes it from the GGUF metadata
        let threads = Int32(max(1, min(8, ProcessInfo.processInfo.activeProcessorCount - 1)))
        params.n_threads = threads
        params.n_threads_batch = threads

        guard let context = llama_init_from_model(model.pointer, params) else {
            throw LlamaError.invalidModel(reason: "a context could not be created (not enough memory?)")
        }
        self.model = model
        self.pointer = context
        self.batchCapacity = batchTokens
        self.maxSequences = maxSequences
        self.batch = llama_batch_init(Int32(batchTokens), 0, Int32(maxSequences))
    }

    deinit {
        llama_batch_free(batch)
        llama_free(pointer)
    }

    var poolingType: llama_pooling_type { llama_pooling_type(pointer) }

    /// Runs the sequences through the model. `sequences[i]` becomes sequence id `i`; every token
    /// requests an output so pooled embeddings are available afterwards.
    func process(_ sequences: [[llama_token]]) throws {
        let total = sequences.reduce(0) { $0 + $1.count }
        guard sequences.count <= maxSequences, total <= batchCapacity, total > 0 else {
            throw LlamaError.backend(operation: "batch size", code: -1)
        }
        var position = 0
        for (sequence, tokens) in sequences.enumerated() {
            for (index, token) in tokens.enumerated() {
                guard let sequenceIDs = batch.seq_id[position] else {
                    throw LlamaError.backend(operation: "batch layout", code: -1)
                }
                batch.token[position] = token
                batch.pos[position] = Int32(index)
                batch.n_seq_id[position] = 1
                sequenceIDs[0] = Int32(sequence)
                batch.logits[position] = 1
                position += 1
            }
        }
        batch.n_tokens = Int32(position)

        if let memory = llama_get_memory(pointer) { llama_memory_clear(memory, true) }
        let code = model.hasEncoder ? llama_encode(pointer, batch) : llama_decode(pointer, batch)
        guard code == 0 else {
            throw LlamaError.backend(operation: model.hasEncoder ? "encode" : "decode", code: code)
        }
    }

    // MARK: Generation

    /// Context size per sequence.
    var contextTokens: Int { Int(llama_n_ctx_seq(pointer)) }

    func clearMemory() {
        if let memory = llama_get_memory(pointer) { llama_memory_clear(memory, true) }
    }

    /// Feeds `tokens` (one sequence, starting at position 0) in batch-sized chunks. Only the final
    /// token produces logits, ready for sampling.
    func decodePrompt(_ tokens: [llama_token], isCancelled: () -> Bool) throws {
        guard !tokens.isEmpty, tokens.count <= contextTokens else {
            throw LlamaError.backend(operation: "prompt size", code: -1)
        }
        var start = 0
        while start < tokens.count {
            if isCancelled() { throw CancellationError() }
            let end = min(start + batchCapacity, tokens.count)
            try fill(tokens[start..<end], firstPosition: start, logitsOnLast: end == tokens.count)
            start = end
        }
    }

    /// Feeds one generated token at `position`.
    func decode(token: llama_token, position: Int) throws {
        try fill([token][...], firstPosition: position, logitsOnLast: true)
    }

    private func fill(_ tokens: ArraySlice<llama_token>, firstPosition: Int, logitsOnLast: Bool) throws {
        for (offset, token) in tokens.enumerated() {
            guard let sequenceIDs = batch.seq_id[offset] else { throw LlamaError.backend(operation: "batch layout", code: -1) }
            batch.token[offset] = token
            batch.pos[offset] = Int32(firstPosition + offset)
            batch.n_seq_id[offset] = 1
            sequenceIDs[0] = 0
            batch.logits[offset] = (logitsOnLast && offset == tokens.count - 1) ? 1 : 0
        }
        batch.n_tokens = Int32(tokens.count)
        let code = llama_decode(pointer, batch)
        guard code == 0 else { throw LlamaError.backend(operation: "decode", code: code) }
    }

    /// The pooled embedding of sequence `sequence` from the last `process` call.
    func pooledEmbedding(sequence: Int, dimension: Int) throws -> [Float] {
        guard let pointer = llama_get_embeddings_seq(pointer, Int32(sequence)) else {
            throw LlamaError.backend(operation: "read embeddings", code: -1)
        }
        return Array(UnsafeBufferPointer(start: pointer, count: dimension))
    }
}
