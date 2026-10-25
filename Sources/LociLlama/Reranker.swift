import Foundation
import llama

/// A GGUF cross-encoder reranker on llama.cpp (rank pooling).
///
/// Two input formats are supported, chosen from the model file:
/// - models that ship a `rerank` chat template (Qwen3-Reranker) are prompted with it and score as
///   the log-odds of `yes` from their two-way classifier head;
/// - classic cross-encoders (bge-reranker-v2-m3) get `BOS query EOS SEP document EOS` and use their single output.
public final class Reranker: TextReranker, @unchecked Sendable {
    // Safety of @unchecked Sendable: `context` is only used inside `LlamaExecutor.run`, which
    // executes on one serial queue. Everything else is an immutable `let`.
    public let id: String
    private let model: LlamaModel
    private let context: LlamaContext
    private let layout: Layout
    private let scoring: Scoring

    /// How a (query, passage) pair is turned into tokens.
    private enum Layout {
        case template(prefix: String, middle: String, suffix: String)  // text before the query, between, after the passage
        case pair
    }

    /// How the classifier outputs become one number.
    private enum Scoring {
        case single
        case yesMinusNo(yes: Int, no: Int)
    }

    private init(model: LlamaModel, context: LlamaContext, id: String, layout: Layout, scoring: Scoring) {
        self.model = model
        self.context = context
        self.id = id
        self.layout = layout
        self.scoring = scoring
    }

    /// Context size; one (query, passage) pair never exceeds it.
    static let contextTokens = 1024

    /// Loads the reranker on the llama queue.
    ///
    /// Throws `LlamaError.roleMismatch` when the model has no classification head, so a chat or
    /// embedding model cannot be used as a reranker.
    public static func load(path: String, gpuLayers: Int32) async throws -> Reranker {
        try await LlamaExecutor.run { _ in
            // Checked from the header first: a chat model is rejected without loading gigabytes of weights.
            let header = try GGUFHeader.read(path: path)
            try header.requireRerankerRole()
            let model = try LlamaModel(path: path, gpuLayers: gpuLayers)
            guard model.classifierOutputCount > 0 else { throw LlamaError.roleMismatch(expected: "reranker") }

            let batchTokens = min(contextTokens, max(8, model.trainingContext))
            // Some conversions omit the pooling type; a classifier head means rank pooling is what they want.
            let context = try LlamaContext(
                model: model, contextTokens: batchTokens, batchTokens: batchTokens, maxSequences: 8,
                embeddings: true, pooling: LLAMA_POOLING_TYPE_RANK, cpuOnly: gpuLayers == 0)
            guard context.poolingType == LLAMA_POOLING_TYPE_RANK else { throw LlamaError.roleMismatch(expected: "reranker") }

            return Reranker(
                model: model, context: context, id: header.identityString(),
                layout: Self.layout(for: model), scoring: try Self.scoring(for: model))
        }
    }

    /// The model identity without loading weights; equals `id` of the loaded reranker.
    public static func identity(ofModelAt path: String) async throws -> String {
        try await LlamaExecutor.run { _ in try GGUFHeader.read(path: path).identityString() }
    }

    /// Estimated memory for loading the reranker at `path`, from its header alone.
    public static func estimatedMemory(modelAt path: String) throws -> UInt64 {
        let header = try GGUFHeader.read(path: path)
        try header.requireRerankerRole()
        return MemoryEstimate.estimate(header: header, contextTokens: contextTokens).total
    }

    private static func layout(for model: LlamaModel) -> Layout {
        guard let template = model.chatTemplate(named: "rerank"),
              let query = template.range(of: "{query}"), let document = template.range(of: "{document}"),
              query.upperBound <= document.lowerBound else { return .pair }
        return .template(
            prefix: String(template[..<query.lowerBound]),
            middle: String(template[query.upperBound..<document.lowerBound]),
            suffix: String(template[document.upperBound...]))
    }

    private static func scoring(for model: LlamaModel) throws -> Scoring {
        switch model.classifierOutputCount {
        case 1:
            return .single
        case 2:
            let labels = (0..<2).map { model.classifierLabel($0)?.lowercased() }
            guard let yes = labels.firstIndex(of: "yes"), let no = labels.firstIndex(of: "no") else {
                throw LlamaError.invalidModel(reason: "the reranker's two outputs are not labelled yes/no")
            }
            return .yesMinusNo(yes: yes, no: no)
        default:
            throw LlamaError.invalidModel(reason: "unsupported number of reranker outputs")
        }
    }

    public func score(query: String, passages: [String], instruction: String?) async throws -> [Float] {
        guard !passages.isEmpty else { return [] }
        return try await LlamaExecutor.run { isCancelled in
            let sequences = passages.map { self.tokens(query: query, passage: $0, instruction: instruction) }
            var scores: [Float] = []
            scores.reserveCapacity(passages.count)

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
                let batch = Array(sequences[start..<end])
                try self.context.process(batch)
                for index in batch.indices {
                    let outputs = try self.context.pooledEmbedding(sequence: index, dimension: self.model.classifierOutputCount)
                    scores.append(self.combine(outputs))
                }
                start = end
            }
            return scores
        }
    }

    private func combine(_ outputs: [Float]) -> Float {
        switch scoring {
        case .single:
            return outputs.first ?? 0
        case .yesMinusNo(let yes, let no):
            let (p, q) = (outputs[yes], outputs[no])
            // llama.cpp softmaxes this head, so the two outputs are probabilities. Their difference
            // saturates at ±1 and stops ordering clear misses; the log-odds does not.
            if outputs.allSatisfy({ (0...1).contains($0) }), abs(p + q - 1) < 1e-3 {
                let floor: Float = 1e-12
                return log(max(p, floor)) - log(max(q, floor))
            }
            return p - q  // raw logits: the margin is already the log-odds
        }
    }

    // MARK: Input construction

    /// The template text with the line `<Instruct>: …` swapped for `instruction`; unchanged if the template has none.
    static func replacingInstruction(in prefix: String, with instruction: String) -> String {
        guard let label = prefix.range(of: "<Instruct>: "), let end = prefix.range(of: "\n<Query>:", range: label.upperBound..<prefix.endIndex) else {
            return prefix
        }
        return prefix.replacingCharacters(in: label.upperBound..<end.lowerBound, with: instruction.replacingOccurrences(of: "\n", with: " "))
    }

    /// Tokens for one pair, truncating the passage (then the query) so the pair fits the context.
    /// Neither the query nor the passage can inject special tokens.
    func tokens(query: String, passage: String, instruction: String? = nil) -> [llama_token] {
        let limit = context.batchCapacity
        switch layout {
        case .pair:
            let overhead = 4  // BOS, EOS, SEP, EOS
            var queryTokens = model.tokenize(query, addSpecial: false)
            var passageTokens = model.tokenize(passage, addSpecial: false)
            if queryTokens.count > limit / 3 { queryTokens = Array(queryTokens.prefix(limit / 3)) }
            let room = max(limit - overhead - queryTokens.count, 1)
            if passageTokens.count > room { passageTokens = Array(passageTokens.prefix(room)) }
            return [model.beginToken] + queryTokens + [model.endToken, model.separatorToken] + passageTokens + [model.endToken]

        case .template(let defaultPrefix, let middle, let suffix):
            let prefix = instruction.map { Self.replacingInstruction(in: defaultPrefix, with: model.removingSpecialTokenTexts($0)) } ?? defaultPrefix
            let fixed = model.tokenize(prefix + middle + suffix, addSpecial: false, parseSpecial: true).count
            var safeQuery = model.removingSpecialTokenTexts(query)
            var safePassage = model.removingSpecialTokenTexts(passage)

            let queryTokens = model.tokenize(safeQuery, addSpecial: false)
            if queryTokens.count > limit / 3 { safeQuery = model.detokenize(Array(queryTokens.prefix(limit / 3))) }
            let room = max(limit - fixed - min(queryTokens.count, limit / 3) - 8, 1)
            let passageTokens = model.tokenize(safePassage, addSpecial: false)
            if passageTokens.count > room { safePassage = model.detokenize(Array(passageTokens.prefix(room))) }

            let text = prefix + safeQuery + middle + safePassage + suffix
            let tokens = model.tokenize(text, addSpecial: false, parseSpecial: true)
            // Joint tokenization can differ from the sum of the parts by a few tokens; cut the passage tail if so.
            return tokens.count <= limit ? tokens : Array(tokens.prefix(limit))
        }
    }
}
