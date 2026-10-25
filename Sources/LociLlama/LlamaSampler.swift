import llama

/// Owns a llama.cpp sampler chain and frees it in `deinit`.
final class LlamaSampler {
    private let chain: UnsafeMutablePointer<llama_sampler>

    /// Greedy decoding at `temperature` 0; otherwise top-k, top-p and temperature sampling.
    /// A fixed `seed` makes sampled output repeatable.
    init?(temperature: Float, seed: UInt32) {
        guard let chain = llama_sampler_chain_init(llama_sampler_chain_default_params()) else { return nil }
        self.chain = chain
        if temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(40))
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1))
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(seed))
        }
    }

    deinit {
        llama_sampler_free(chain)
    }

    /// Samples from the last decoded position and records the token.
    func next(in context: LlamaContext) -> llama_token {
        llama_sampler_sample(chain, context.pointer, -1)
    }
}
