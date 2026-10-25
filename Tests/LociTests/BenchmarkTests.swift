import Foundation
import Testing
@testable import Loci
@testable import LociEval

@Suite("Synthetic corpus")
struct SyntheticCorpusTests {
    @Test func isDeterministicForASeedAndDifferentAcrossSeeds() {
        #expect(SyntheticCorpus(seed: 1).document(chunks: 20) == SyntheticCorpus(seed: 1).document(chunks: 20))
        #expect(SyntheticCorpus(seed: 1).document(chunks: 20) != SyntheticCorpus(seed: 2).document(chunks: 20))
        #expect(SyntheticCorpus(seed: 1).queries(10) == SyntheticCorpus(seed: 1).queries(10))
    }

    @Test func hasTheRequestedNumberOfDistinctParagraphs() {
        let paragraphs = SyntheticCorpus(seed: 5).document(chunks: 500).components(separatedBy: "\n\n")
        #expect(paragraphs.count == 500 && Set(paragraphs).count == 500)
        #expect(paragraphs.allSatisfy { $0.split(separator: " ").count > 30 && $0.split(separator: " ").count < 50 })
    }

    @Test func queriesAreSixWordsAndNeverEmpty() {
        let queries = SyntheticCorpus(seed: 9).queries(25)
        #expect(queries.count == 25 && queries.allSatisfy { $0.split(separator: " ").count == 6 })
        #expect(SyntheticCorpus(seed: 9).queries(0).count == 1, "at least one query so timings are never empty")
    }
}

@Suite("Benchmark")
struct BenchmarkTests {
    private func makeBenchmark(reranker: Bool = true, generator: Bool = false) -> Benchmark {
        Benchmark {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("loci-bench-test-\(UUID().uuidString)")
            let services = await ModelServices(
                embedder: FakeEmbedder(), reranker: reranker ? FakeReranker() : nil,
                generator: generator ? FakeGenerator("The lighthouse tower is 38 metres tall [1].") : nil)
            return try await KnowledgeBase(directory: directory, services: services)
        }
    }

    @Test func timesIngestEverySearchModeAndTheColdStarts() async throws {
        let report = try await makeBenchmark().run(BenchmarkOptions(chunkCount: 300, queries: 12, rerankQueries: 3))

        #expect(report.ingest.chunks >= 250, "about one passage per paragraph: \(report.ingest.chunks)")
        #expect(report.ingest.seconds > 0 && report.ingest.chunksPerSecond > 0 && report.ingest.indexBytes > 0)
        #expect(report.search.map(\.name) == ["keyword search", "vector search", "hybrid search", "hybrid + rerank search (20 candidates reranked)"])
        #expect(report.search.map(\.samples) == [12, 12, 12, 3])
        for timing in report.search {
            #expect(timing.p50Ms <= timing.p95Ms && timing.p95Ms <= timing.maxMs && timing.meanMs > 0, "\(timing)")
        }
        #expect(report.coldStarts.count == 2 && report.coldStarts.allSatisfy { $0.samples == 1 && $0.meanMs > 0 })
        #expect(report.coldStarts[0].name.contains("vector snapshot"))
        #expect(report.answers == nil)
        #expect(report.memory.map(\.stage).first == "start" && report.memory.count >= 3)
        #expect(report.peakFootprintBytes > 0)
        #expect(report.environment.reranker == "fake-reranker|1|0")
    }

    @Test func withoutARerankerThereIsNoRerankedRow() async throws {
        let report = try await makeBenchmark(reranker: false).run(BenchmarkOptions(chunkCount: 100, queries: 5))
        #expect(report.search.count == 3 && report.coldStarts.count == 1)
    }

    @Test func timesAnswersWhenGivenADatasetAndAGenerator() async throws {
        let dataset = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/eval-small", isDirectory: true)
        let report = try await makeBenchmark(generator: true).run(BenchmarkOptions(chunkCount: 100, queries: 5, rerankQueries: 2, answerDataset: dataset))
        let performance = try #require(report.answers)
        #expect(performance.totalP50Ms > 0 && performance.tokensPerSecondMean > 0)
        #expect(report.memory.last?.stage == "after answers")
    }

    @Test func theMarkdownCarriesEverySectionAndTheCaveats() async throws {
        let report = try await makeBenchmark().run(BenchmarkOptions(chunkCount: 100, queries: 5, rerankQueries: 2))
        let markdown = Benchmark.markdown(report)
        for heading in ["# Benchmarks", "## Ingest", "## Search latency over", "## Cold starts", "## Memory"] { #expect(markdown.contains(heading), Comment(rawValue: heading)) }
        #expect(markdown.contains("not from anywhere else") && markdown.contains("have not been measured"))
        #expect(markdown.contains("| vector search |") && markdown.contains("include embedding the query"))
        #expect(!markdown.contains("\\("), "no unexpanded interpolation")
        let data = try JSONEncoder().encode(report)
        #expect(try JSONDecoder().decode(BenchmarkReport.self, from: data) == report)
    }
}
