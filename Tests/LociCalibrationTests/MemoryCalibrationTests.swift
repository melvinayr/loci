import Foundation
import Testing
@testable import LociLlama

/// Where the models live. This target exists on its own so these tests run alone: they measure the whole
/// process's memory, and any model another test loads at the same time would inflate the numbers.
enum Models {
    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let fixtureEmbedder = root.appendingPathComponent("Tests/Fixtures/models/bge-small-en-v1.5-q8_0.gguf")
    static let fixtureReranker = root.appendingPathComponent("Tests/Fixtures/models/qwen3-reranker-0.6b-q8_0.gguf")
    static let localEmbedder = root.appendingPathComponent("LLM/embeddinggemma-300M-Q8_0.gguf")
    static let generator = root.appendingPathComponent("LLM/gemma-4-E4B-it-Q4_0.gguf")
    static func exists(_ url: URL) -> Bool { FileManager.default.fileExists(atPath: url.path) }
}

/// The estimator must not under-report what loading really costs, or `.insufficientMemory` would not protect anyone.
@Suite("Memory estimates against measured footprint", .serialized)
struct MemoryEstimateCalibrationTests {
    private func measuredGrowth(_ load: () async throws -> AnyObject) async throws -> UInt64 {
        var smallest = UInt64.max
        for _ in 0..<2 {
            let before = SystemMemoryProbe.currentFootprint()
            let model = try await load()
            let after = SystemMemoryProbe.currentFootprint()
            withExtendedLifetime(model) {}
            smallest = min(smallest, after > before ? after - before : 0)
        }
        return smallest
    }

    @Test(.enabled(if: Models.exists(Models.fixtureEmbedder)))
    func embedderEstimateCoversMeasuredGrowth() async throws {
        let path = Models.fixtureEmbedder.path
        let estimate = try Embedder.estimatedMemory(modelAt: path)
        let measured = try await measuredGrowth {
            let embedder = try await Embedder.load(path: path, gpuLayers: 0)
            _ = try await embedder.embed(Array(repeating: String(repeating: "word ", count: 200), count: 8), role: .document)
            return embedder
        }
        #expect(Double(estimate) >= Double(measured) * 0.8, "estimate \(estimate) vs measured \(measured)")
        // An absurdly high estimate would refuse loads that fit. (Measured growth is no use as an upper bound:
        // pages freed by an earlier model get reused, so growth can look tiny in a busy process.)
        let fileSize = try #require((try FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?.uint64Value)
        #expect(estimate <= 3 * fileSize + (128 << 20), "estimate \(estimate) for a \(fileSize)-byte file")
    }

    @Test(.enabled(if: Models.exists(Models.fixtureReranker)))
    func rerankerEstimateCoversMeasuredGrowth() async throws {
        let path = Models.fixtureReranker.path
        let estimate = try Reranker.estimatedMemory(modelAt: path)
        let measured = try await measuredGrowth {
            let reranker = try await Reranker.load(path: path, gpuLayers: 0)
            _ = try await reranker.score(query: "q", passages: Array(repeating: String(repeating: "word ", count: 100), count: 4))
            return reranker
        }
        #expect(Double(estimate) >= Double(measured) * 0.8, "estimate \(estimate) vs measured \(measured)")
        let fileSize = try #require((try FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?.uint64Value)
        #expect(estimate <= 3 * fileSize + (128 << 20), "estimate \(estimate) for a \(fileSize)-byte file")
    }

    @Test(.enabled(if: Models.exists(Models.localEmbedder)))
    func aLargerEmbedderWithAKVCacheIsCoveredToo() async throws {
        let path = Models.localEmbedder.path
        let estimate = try Embedder.estimatedMemory(modelAt: path)
        let measured = try await measuredGrowth {
            let embedder = try await Embedder.load(path: path, gpuLayers: 0)
            _ = try await embedder.embed(Array(repeating: String(repeating: "word ", count: 200), count: 8), role: .document)
            return embedder
        }
        #expect(Double(estimate) >= Double(measured) * 0.8, "estimate \(estimate) vs measured \(measured)")
    }

    @Test(.enabled(if: Models.exists(Models.generator)))
    func aChatModelCanBeSizedWithoutLoadingIt() throws {
        let estimate = try MemoryEstimate.estimate(modelAt: Models.generator.path, contextTokens: 4096)
        #expect(estimate.weights > 4_000_000_000)
        #expect(estimate.kvCache > 0, "a decoder model needs a KV cache")
    }
}
