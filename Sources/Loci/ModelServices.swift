import Foundation
internal import LociIngest
internal import LociLlama

/// Resolves a `ModelSource` to a file on disk.
///
/// `.hub` models are looked up in the folder `LociHub` downloads into; this target never
/// touches the network.
enum ModelResolver {
    static func url(for source: ModelSource) throws -> URL {
        let candidate: URL?
        switch source {
        case .file(let url):
            candidate = url
        case .bundle(let name, let bundle):
            candidate = bundle.url(forResource: name, withExtension: nil)
                ?? bundle.url(forResource: name, withExtension: "gguf")
        case .hub(_, let file, _):
            candidate = modelsDirectory?.appendingPathComponent(file)
        }
        guard let candidate, FileManager.default.fileExists(atPath: candidate.path) else {
            throw LociError.modelNotFound(source)
        }
        return candidate
    }

    /// Application Support/Loci/Models, where `LociHub` stores downloads.
    static var modelsDirectory: URL? {
        try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            .appendingPathComponent("Loci", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }
}

/// Boxes so protocol-typed models can be held by `ModelResidencyManager`, which needs class references.
final class EmbedderBox: Sendable {
    let value: any TextEmbedder
    init(_ value: any TextEmbedder) { self.value = value }
}

final class RerankerBox: Sendable {
    let value: any TextReranker
    init(_ value: any TextReranker) { self.value = value }
}

final class GeneratorBox: Sendable {
    let value: any TextGenerator
    init(_ value: any TextGenerator) { self.value = value }
}

extension ModelResidency {
    var policy: ResidencyPolicy {
        switch self {
        case .lowMemory: .lowMemory
        case .balanced: .balanced
        case .keepResident: .keepResident
        }
    }
}

/// The models a knowledge base uses, loaded lazily and kept or released according to its
/// residency policy.
actor ModelServices {
    let manager: ModelResidencyManager
    private let readEmbedderIdentity: @Sendable () async throws -> EmbedderIdentity
    private let readRerankerID: (@Sendable () async throws -> String)?
    private let readGeneratorID: (@Sendable () async throws -> String)?
    private var cachedEmbedderIdentity: EmbedderIdentity?

    var hasReranker: Bool { readRerankerID != nil }
    var hasGenerator: Bool { readGeneratorID != nil }

    /// Models from a `ModelSet`, loaded with llama.cpp on first use.
    init(models: ModelSet, options: KnowledgeBaseOptions) async {
        let layers: Int32 = switch options.gpuLayers {
        case .all: 999
        case .none: 0
        case .count(let count): Int32(clamping: max(0, count))
        }
        manager = ModelResidencyManager(policy: options.residency.policy)

        let embedderSource = models.embedder
        readEmbedderIdentity = {
            let path = try ModelResolver.url(for: embedderSource).path
            do { return try await Embedder.identity(ofModelAt: path) } catch { throw LociError.from(error, model: embedderSource) }
        }
        await manager.register(
            .embedder,
            estimatedBytes: {
                let path = try ModelResolver.url(for: embedderSource).path
                do { return try Embedder.estimatedMemory(modelAt: path) } catch { throw LociError.from(error, model: embedderSource) }
            },
            load: {
                let path = try ModelResolver.url(for: embedderSource).path
                do { return EmbedderBox(try await Embedder.load(path: path, gpuLayers: layers)) } catch {
                    throw LociError.from(error, model: embedderSource)
                }
            })

        if let rerankerSource = models.reranker {
            readRerankerID = {
                let path = try ModelResolver.url(for: rerankerSource).path
                do { return try await Reranker.identity(ofModelAt: path) } catch { throw LociError.from(error, model: rerankerSource) }
            }
            await manager.register(
                .reranker,
                estimatedBytes: {
                    let path = try ModelResolver.url(for: rerankerSource).path
                    do { return try Reranker.estimatedMemory(modelAt: path) } catch { throw LociError.from(error, model: rerankerSource) }
                },
                load: {
                    let path = try ModelResolver.url(for: rerankerSource).path
                    do { return RerankerBox(try await Reranker.load(path: path, gpuLayers: layers)) } catch {
                        throw LociError.from(error, model: rerankerSource)
                    }
                })
        } else {
            readRerankerID = nil
        }

        if let generatorSource = models.generator {
            readGeneratorID = {
                let path = try ModelResolver.url(for: generatorSource).path
                do { return try await Generator.identity(ofModelAt: path) } catch { throw LociError.from(error, model: generatorSource) }
            }
            await manager.register(
                .generator,
                estimatedBytes: {
                    let path = try ModelResolver.url(for: generatorSource).path
                    do { return try Generator.estimatedMemory(modelAt: path) } catch { throw LociError.from(error, model: generatorSource) }
                },
                load: {
                    let path = try ModelResolver.url(for: generatorSource).path
                    do { return GeneratorBox(try await Generator.load(path: path, gpuLayers: layers)) } catch {
                        throw LociError.from(error, model: generatorSource)
                    }
                })
        } else {
            readGeneratorID = nil
        }
    }

    /// Ready-made models, for tests. `estimatedBytes` is what each model claims to need.
    init(
        embedder: any TextEmbedder, reranker: (any TextReranker)? = nil, generator: (any TextGenerator)? = nil,
        policy: ResidencyPolicy = .keepResident,
        memory: any MemoryProbe = SystemMemoryProbe(), estimatedBytes: UInt64 = 0
    ) async {
        manager = ModelResidencyManager(policy: policy, memory: memory, observesMemoryPressure: false)
        readEmbedderIdentity = { embedder.identity }
        await manager.register(.embedder, estimatedBytes: { estimatedBytes }, load: { EmbedderBox(embedder) })
        if let reranker {
            readRerankerID = { reranker.id }
            await manager.register(.reranker, estimatedBytes: { estimatedBytes }, load: { RerankerBox(reranker) })
        } else {
            readRerankerID = nil
        }
        if let generator {
            readGeneratorID = { "test-generator" }
            await manager.register(.generator, estimatedBytes: { estimatedBytes }, load: { GeneratorBox(generator) })
        } else {
            readGeneratorID = nil
        }
    }

    /// The embedder's identity, read from the model file without loading it.
    func embedderIdentity() async throws -> EmbedderIdentity {
        if let cachedEmbedderIdentity { return cachedEmbedderIdentity }
        let identity = try await readEmbedderIdentity()
        cachedEmbedderIdentity = identity
        return identity
    }

    func rerankerID() async throws -> String? {
        try await readRerankerID?()
    }

    /// Runs `body` with the embedder loaded; it cannot be unloaded until `body` returns.
    func withEmbedder<Result: Sendable>(_ body: @Sendable (any TextEmbedder) async throws -> Result) async throws -> Result {
        try await manager.use(.embedder, as: EmbedderBox.self) { try await body($0.value) }
    }

    /// One relevance score per passage, loading the reranker if needed.
    func rerank(query: String, passages: [String], instruction: String? = nil) async throws -> [Float] {
        try await manager.use(.reranker, as: RerankerBox.self) {
            try await $0.value.score(query: query, passages: passages, instruction: instruction)
        }
    }

    func generatorID() async throws -> String? {
        try await readGeneratorID?()
    }

    /// Runs `body` with the generator loaded; it cannot be unloaded until `body` returns.
    func withGenerator<Result: Sendable>(_ body: @Sendable (any TextGenerator) async throws -> Result) async throws -> Result {
        try await manager.use(.generator, as: GeneratorBox.self) { try await body($0.value) }
    }

    func unloadAll() async {
        await manager.unloadAll()
    }
}

/// Adapts the embedder's tokenizer to the chunker.
struct EmbedderTokenCounter: TokenCounting {
    let embedder: any TextEmbedder

    func count(_ scalars: ArraySlice<Unicode.Scalar>) -> Int {
        embedder.countTokens(scalars)
    }
}
