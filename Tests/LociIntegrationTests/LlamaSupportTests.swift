import Foundation
import Testing
import llama
@testable import LociLlama

/// M0 verification: the pinned llama.cpp release supports embedding and rank pooling,
/// and can read the model architectures we intend to use. Throwaway C-API probing is
/// confined to tests; production code lives in `LociLlama`.
@Suite("Pinned llama.cpp release")
struct LlamaSupportTests {
    @Test func exposesPoolingTypesNeededForEmbeddingAndReranking() {
        #expect(LLAMA_POOLING_TYPE_MEAN.rawValue == 1)
        #expect(LLAMA_POOLING_TYPE_CLS.rawValue == 2)
        #expect(LLAMA_POOLING_TYPE_LAST.rawValue == 3)
        #expect(LLAMA_POOLING_TYPE_RANK.rawValue == 4)
    }

    @Test(.enabled(if: ModelFixtures.exists(ModelFixtures.embedder)))
    func embedderLoadsPoolsAndReturnsUnitScaleVector() throws {
        LlamaBackend.ensureInitialized()

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = 0
        let model = try #require(llama_model_load_from_file(ModelFixtures.embedder.path, modelParams))
        defer { llama_model_free(model) }

        var contextParams = llama_context_default_params()
        contextParams.n_ctx = 512
        contextParams.n_batch = 512
        contextParams.n_ubatch = 512
        contextParams.embeddings = true
        contextParams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED  // take it from GGUF metadata
        let context = try #require(llama_init_from_model(model, contextParams))
        defer { llama_free(context) }

        // Pooling must resolve from metadata to something other than NONE/UNSPECIFIED.
        let pooling = llama_pooling_type(context)
        #expect(pooling == LLAMA_POOLING_TYPE_MEAN || pooling == LLAMA_POOLING_TYPE_CLS || pooling == LLAMA_POOLING_TYPE_LAST)

        let dimension = Int(llama_model_n_embd_out(model))
        #expect(dimension > 0)

        let vocab = try #require(llama_model_get_vocab(model))
        let text = "the method of loci"
        var tokens = [llama_token](repeating: 0, count: 64)
        let count = llama_tokenize(vocab, text, Int32(text.utf8.count), &tokens, Int32(tokens.count), true, false)
        #expect(count > 0)

        var batch = llama_batch_init(512, 0, 1)
        defer { llama_batch_free(batch) }
        batch.n_tokens = count
        for i in 0..<Int(count) {
            batch.token[i] = tokens[i]
            batch.pos[i] = Int32(i)
            batch.n_seq_id[i] = 1
            batch.seq_id[i]![0] = 0
            batch.logits[i] = 1
        }
        let status = llama_decode(context, batch)
        #expect(status == 0)

        let pointer = try #require(llama_get_embeddings_seq(context, 0))
        let vector = Array(UnsafeBufferPointer(start: pointer, count: dimension))
        #expect(vector.allSatisfy { $0.isFinite })
        let norm = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
        #expect(norm > 0)
    }

    @Test(.enabled(if: ModelFixtures.exists(ModelFixtures.generator)))
    func generatorArchitectureIsRecognised() throws {
        LlamaBackend.ensureInitialized()
        var params = llama_model_default_params()
        params.vocab_only = true  // metadata + tokenizer only; avoids loading 4 GB of weights
        let model = try #require(llama_model_load_from_file(ModelFixtures.generator.path, params))
        defer { llama_model_free(model) }
        #expect(llama_model_get_vocab(model) != nil)
    }
}
