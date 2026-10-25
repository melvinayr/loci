import LociLlama

/// The conversation for one answer and what went into it.
public struct PackedContext: Sendable {
    public let messages: [ChatMessage]
    /// How many of the candidate sources were included, always the best-ranked ones.
    public let sourceCount: Int
    public let promptTokens: Int
}

/// Fits as many of the best-ranked sources as the budget allows.
public enum ContextPacker {
    /// Includes sources `1…k` for the largest `k` whose rendered prompt fits in `maxTokens`. Lower-ranked
    /// sources are dropped first and a source is never cut short. Returns nil when even the first source
    /// does not fit.
    ///
    /// `countTokens` must measure the real rendered conversation (template included), so the budget
    /// is exact for the model in use.
    public static func pack(
        question: String, sources: [PromptSource], instructions: String?, maxTokens: Int,
        sanitize: (String) -> String, countTokens: ([ChatMessage]) async throws -> Int
    ) async throws -> PackedContext? {
        guard !sources.isEmpty else { return nil }
        func messages(_ count: Int) -> [ChatMessage] {
            GroundingPrompt.messages(
                question: question, sources: Array(sources.prefix(count)), instructions: instructions, sanitize: sanitize)
        }

        // Prompt size grows with every added source, so the largest fitting prefix can be found by bisection.
        var low = 0, high = sources.count  // `low` is known to fit (or is 0), answers lie in low...high
        var best: (count: Int, tokens: Int)?
        while low < high {
            let middle = (low + high + 1) / 2
            let tokens = try await countTokens(messages(middle))
            if tokens <= maxTokens {
                best = (middle, tokens)
                low = middle
            } else {
                high = middle - 1
            }
        }
        guard let best else { return nil }
        return PackedContext(messages: messages(best.count), sourceCount: best.count, promptTokens: best.tokens)
    }
}
