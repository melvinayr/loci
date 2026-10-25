import Foundation
import llama

public struct GenerationParameters: Sendable, Equatable {
    public var maxTokens: Int
    /// 0 decodes greedily.
    public var temperature: Float
    /// Fixes sampling randomness for repeatable output; nil picks a fresh seed.
    public var seed: UInt32?

    public init(maxTokens: Int = 400, temperature: Float = 0.2, seed: UInt32? = nil) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.seed = seed
    }
}

public struct GenerationStats: Sendable, Equatable {
    public enum StopReason: Sendable, Equatable { case endOfText, maxTokens, contextFull }

    public var promptTokens: Int
    public var generatedTokens: Int
    /// From the start of the request, including prompt processing, to the first visible text.
    public var timeToFirstToken: Duration
    /// Tokens per second after the first, excluding prompt processing.
    public var tokensPerSecond: Double
    public var stopReason: StopReason

    public init(
        promptTokens: Int, generatedTokens: Int, timeToFirstToken: Duration, tokensPerSecond: Double, stopReason: StopReason
    ) {
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.timeToFirstToken = timeToFirstToken
        self.tokensPerSecond = tokensPerSecond
        self.stopReason = stopReason
    }
}

public enum GenerationEvent: Sendable, Equatable {
    /// Visible text: never a partial code point, and with reasoning blocks removed.
    case text(String)
    case finished(GenerationStats)
}

/// Produces replies from a chat model. `Generator` is the llama.cpp implementation; tests
/// substitute scripted fakes.
public protocol TextGenerator: Sendable {
    /// Tokens the model can hold in one conversation (prompt plus reply).
    var contextTokens: Int { get }
    /// Tokens the rendered conversation occupies, using the model's own template and tokenizer.
    func promptTokenCount(_ messages: [ChatMessage]) async throws -> Int
    /// `text` with anything the model could read as chat markup (special tokens) removed. Apply it to
    /// every piece of untrusted text, such as documents and questions, before building messages.
    func sanitize(_ text: String) -> String
    /// Streams the reply. Cancelling the consumer stops generation.
    func generate(_ messages: [ChatMessage], parameters: GenerationParameters) -> AsyncThrowingStream<GenerationEvent, Error>
}

/// A GGUF chat model on llama.cpp.
public final class Generator: TextGenerator, @unchecked Sendable {
    // Safety of @unchecked Sendable: `context` is only used inside `LlamaExecutor.run`, which executes
    // on one serial queue. Everything else is an immutable `let` or the read-only `LlamaModel`.
    private let model: LlamaModel
    private let context: LlamaContext
    private let template: String
    private let bosText: String
    public let contextTokens: Int

    /// Context size used when the model allows it.
    public static let defaultContextTokens = 4096

    private init(model: LlamaModel, context: LlamaContext, template: String) {
        self.model = model
        self.context = context
        self.template = template
        self.bosText = model.beginText
        self.contextTokens = context.contextTokens
    }

    /// Loads the model on the llama queue.
    ///
    /// Throws `LlamaError.roleMismatch(expected: "generator")` for rerankers, encoder-only models
    /// and models without a chat template, and `invalidModel` when the template is a layout Loci
    /// cannot render.
    public static func load(path: String, gpuLayers: Int32, contextTokens: Int = defaultContextTokens) async throws -> Generator {
        try await LlamaExecutor.run { _ in
            try GGUFHeader.read(path: path).requireGeneratorRole()
            let model = try LlamaModel(path: path, gpuLayers: gpuLayers)
            guard let template = model.chatTemplate(named: nil) else { throw LlamaError.roleMismatch(expected: "generator") }
            do {
                _ = try ChatTemplate.render([ChatMessage(role: .user, content: "test")], template: template, bos: model.beginText)
            } catch {
                throw LlamaError.invalidModel(reason: "the model's chat template is not a layout Loci can render")
            }

            let size = max(256, min(contextTokens, model.trainingContext))
            let context = try LlamaContext(
                model: model, contextTokens: size, batchTokens: min(512, size), maxSequences: 1, embeddings: false,
                cpuOnly: gpuLayers == 0)
            return Generator(model: model, context: context, template: template)
        }
    }

    /// Estimated memory for loading the generator at `path`, from its header alone.
    public static func estimatedMemory(modelAt path: String, contextTokens: Int = defaultContextTokens) throws -> UInt64 {
        let header = try GGUFHeader.read(path: path)
        try header.requireGeneratorRole()
        return MemoryEstimate.estimate(header: header, contextTokens: contextTokens).total
    }

    /// The model identity without loading weights.
    public static func identity(ofModelAt path: String) async throws -> String {
        try await LlamaExecutor.run { _ in try GGUFHeader.read(path: path).identityString() }
    }

    public func sanitize(_ text: String) -> String {
        model.removingSpecialTokenTexts(text)
    }

    public func promptTokenCount(_ messages: [ChatMessage]) async throws -> Int {
        try await LlamaExecutor.run { _ in try self.promptTokens(for: messages).count }
    }

    private func promptTokens(for messages: [ChatMessage]) throws -> [llama_token] {
        let prompt: String
        do {
            prompt = try ChatTemplate.render(messages, template: template, bos: bosText)
        } catch {
            throw LlamaError.invalidModel(reason: "the model's chat template is not a layout Loci can render")
        }
        // The template may already begin with the BOS text; do not add a second one.
        let addBOS = model.addsBeginToken && !(bosText.isEmpty == false && prompt.hasPrefix(bosText))
        return model.tokenize(prompt, addSpecial: addBOS, parseSpecial: true)
    }

    public func generate(_ messages: [ChatMessage], parameters: GenerationParameters) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let cancelled = CancellationFlag()
            let task = Task {
                do {
                    try await LlamaExecutor.run { isCancelledByTask in
                        try self.run(messages, parameters, continuation: continuation) { cancelled.isSet || isCancelledByTask() }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                cancelled.set()
                task.cancel()
            }
        }
    }

    /// The generation loop. Runs on the llama queue.
    private func run(
        _ messages: [ChatMessage], _ parameters: GenerationParameters,
        continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation, isCancelled: () -> Bool
    ) throws {
        let start = ContinuousClock.now
        let prompt = try promptTokens(for: messages)
        guard prompt.count < context.contextTokens else {
            throw LlamaError.backend(operation: "prompt larger than the context", code: -1)
        }
        let budget = min(max(parameters.maxTokens, 1), context.contextTokens - prompt.count)
        guard let sampler = LlamaSampler(temperature: parameters.temperature, seed: parameters.seed ?? UInt32.random(in: 0...UInt32.max)) else {
            throw LlamaError.backend(operation: "create sampler", code: -1)
        }

        context.clearMemory()
        try context.decodePrompt(prompt, isCancelled: isCancelled)

        var decoder = UTF8StreamDecoder()
        var filter = ThinkingFilter()
        var generated = 0
        var firstText: ContinuousClock.Instant?
        var stop = GenerationStats.StopReason.maxTokens

        func emit(_ text: String) {
            guard !text.isEmpty else { return }
            if firstText == nil { firstText = ContinuousClock.now }
            continuation.yield(.text(text))
        }

        while generated < budget {
            if isCancelled() { throw CancellationError() }
            let token = sampler.next(in: context)
            if model.isEndOfGeneration(token) {
                stop = .endOfText
                break
            }
            generated += 1
            emit(filter.feed(decoder.decode(model.pieceBytes(token))))
            if generated == budget { break }
            try context.decode(token: token, position: prompt.count + generated - 1)
        }
        if generated == budget, stop != .endOfText { stop = budget < parameters.maxTokens ? .contextFull : .maxTokens }
        emit(filter.feed(decoder.finish()))
        emit(filter.finish())

        let end = ContinuousClock.now
        let afterFirst = firstText.map { Self.seconds(end - $0) } ?? 0
        continuation.yield(.finished(GenerationStats(
            promptTokens: prompt.count, generatedTokens: generated,
            timeToFirstToken: firstText.map { $0 - start } ?? end - start,
            tokensPerSecond: afterFirst > 0 ? Double(max(generated - 1, 0)) / afterFirst : 0,
            stopReason: stop)))
    }

    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}
