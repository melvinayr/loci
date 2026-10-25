import Foundation
import Testing
@testable import LociLlama

private func cosine(_ a: [Float], _ b: [Float]) -> Float {
    zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
}

/// Embedding models available for this run: the small CI fixture and, locally, `LLM/`.
private let embedderPaths: [String] = {
    [ModelFixtures.fixtureEmbedder, ModelFixtures.localEmbedder]
        .filter(ModelFixtures.exists)
        .map(\.path)
}()

@Suite("Embedder (llama.cpp)", .enabled(if: !embedderPaths.isEmpty))
struct EmbedderTests {
    @Test(arguments: embedderPaths)
    func producesUnitVectorsOfTheAdvertisedDimension(path: String) async throws {
        let embedder = try await Embedder.load(path: path, gpuLayers: 0)
        let vectors = try await embedder.embed(["The method of loci is a memory palace technique."], role: .document)
        #expect(vectors.count == 1)
        #expect(vectors[0].count == embedder.identity.dimension)
        #expect(vectors[0].allSatisfy { $0.isFinite })
        #expect(abs(cosine(vectors[0], vectors[0]) - 1) < 1e-3)
    }

    @Test(arguments: embedderPaths)
    func relatedTextIsCloserThanUnrelatedText(path: String) async throws {
        let embedder = try await Embedder.load(path: path, gpuLayers: 0)
        let query = try await embedder.embed(["How do I bake sourdough bread?"], role: .query)[0]
        let documents = try await embedder.embed([
            "To make sourdough, mix flour, water and starter, let the dough rise overnight, then bake in a hot oven.",
            "The lighthouse at the headland was built in 1874 and automated in 1961.",
        ], role: .document)
        #expect(cosine(query, documents[0]) > cosine(query, documents[1]))
    }

    @Test(arguments: embedderPaths)
    func batchingMatchesEmbeddingOneAtATime(path: String) async throws {
        let embedder = try await Embedder.load(path: path, gpuLayers: 0)
        // 19 texts: more than one batch of 8 sequences.
        let texts = (0..<19).map { "Passage number \($0) talks about subject \($0 * 7 % 5) in some detail." }
        let batched = try await embedder.embed(texts, role: .document)
        #expect(batched.count == texts.count)
        for index in [0, 7, 8, 18] {
            let single = try await embedder.embed([texts[index]], role: .document)[0]
            #expect(cosine(batched[index], single) > 0.999, "text \(index) differs when batched")
        }
    }

    @Test(arguments: embedderPaths)
    func handlesEmptyLongAndUnicodeInput(path: String) async throws {
        let embedder = try await Embedder.load(path: path, gpuLayers: 0)
        let long = String(repeating: "word ", count: 5_000)
        let vectors = try await embedder.embed(["", long, "日本語 😀 café"], role: .document)
        #expect(vectors.count == 3)
        for vector in vectors {
            #expect(vector.count == embedder.identity.dimension && vector.allSatisfy { $0.isFinite })
        }
        #expect(try await embedder.embed([], role: .query).isEmpty)
    }

    @Test(arguments: embedderPaths)
    func identityIsStableAndReadableWithoutLoadingWeights(path: String) async throws {
        let embedder = try await Embedder.load(path: path, gpuLayers: 0)
        let cheap = try await Embedder.identity(ofModelAt: path)
        #expect(cheap.id == embedder.identity.id)
        #expect(cheap.dimension > 0)
        let size = try #require((try FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?.uint64Value)
        #expect(cheap.id.hasSuffix("|\(cheap.dimension)|\(size)"))
    }

    @Test(arguments: embedderPaths)
    func tokenCountsGrowWithLength(path: String) async throws {
        let embedder = try await Embedder.load(path: path, gpuLayers: 0)
        func count(_ s: String) -> Int { embedder.countTokens(ArraySlice(Array(s.unicodeScalars))) }
        #expect(count("") == 0)
        #expect(count("hello") >= 1)
        #expect(count("hello world, this is a longer sentence.") > count("hello"))
    }

    @Test(arguments: embedderPaths)
    func cancellationStopsEmbedding(path: String) async throws {
        let embedder = try await Embedder.load(path: path, gpuLayers: 0)
        let texts = (0..<400).map { "Passage \($0): " + String(repeating: "filler words here ", count: 40) }
        let task = Task { try await embedder.embed(texts, role: .document) }
        try await Task.sleep(for: .milliseconds(20))
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
    }

    @Test(arguments: embedderPaths)
    func missingFileAndGarbageFileAreReported(path: String) async throws {
        await #expect(throws: LlamaError.fileNotFound) { try await Embedder.load(path: path + ".nope", gpuLayers: 0) }
        let junk = FileManager.default.temporaryDirectory.appendingPathComponent("junk-\(UUID().uuidString).gguf")
        try Data(repeating: 7, count: 1024).write(to: junk)
        defer { try? FileManager.default.removeItem(at: junk) }
        await #expect(throws: LlamaError.self) { try await Embedder.load(path: junk.path, gpuLayers: 0) }
    }

    @Test(.enabled(if: ModelFixtures.exists(ModelFixtures.generator)))
    func aChatModelIsRejectedAsAnEmbedder() async throws {
        await #expect(throws: LlamaError.roleMismatch(expected: "embedding")) {
            try await Embedder.load(path: ModelFixtures.generator.path, gpuLayers: 0)
        }
    }
}

@Suite("Embedding prompts")
struct EmbeddingPromptTests {
    @Test func knownModelFamiliesGetTheirTrainingPrefixes() {
        #expect(EmbeddingPrompts.forModel(name: "embeddinggemma-300m", architecture: "gemma-embedding").query.hasPrefix("task: search result"))
        #expect(EmbeddingPrompts.forModel(name: "bge-small-en-v1.5", architecture: "bert").query.hasPrefix("Represent this sentence"))
        #expect(EmbeddingPrompts.forModel(name: "nomic-embed-text-v1.5", architecture: "nomic-bert").document == "search_document: ")
        #expect(EmbeddingPrompts.forModel(name: "multilingual-e5-small", architecture: "bert").document == "passage: ")
        #expect(EmbeddingPrompts.forModel(name: "unknown", architecture: "x") == EmbeddingPrompts())
    }
}
