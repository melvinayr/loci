import Foundation
import Testing
@testable import Loci
@testable import LociEval
@testable import LociIngest

private let evalSmall = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Fixtures/eval-small", isDirectory: true)

@Suite("Retrieval metrics")
struct MetricsTests {
    @Test func hitAtK() {
        let relevance = [false, false, true, false]
        #expect(Metrics.hit(relevance, atK: 1) == false)
        #expect(Metrics.hit(relevance, atK: 2) == false)
        #expect(Metrics.hit(relevance, atK: 3))
        #expect(Metrics.hit([], atK: 5) == false)
    }

    @Test func reciprocalRank() {
        #expect(Metrics.reciprocalRank([true]) == 1)
        #expect(Metrics.reciprocalRank([false, false, true]) == 1.0 / 3)
        #expect(Metrics.reciprocalRank([false, false]) == 0)
        #expect(Metrics.firstRank([false, true, true]) == 2)
        #expect(Metrics.firstRank([]) == nil)
    }

    @Test func ndcgAgainstHandComputedValues() {
        // One relevant passage at rank 1: perfect.
        #expect(Metrics.ndcg([true, false], relevantCount: 1, k: 10) == 1)
        // One relevant passage at rank 3: 1/log2(4) = 0.5, ideal 1.
        #expect(abs(Metrics.ndcg([false, false, true], relevantCount: 1, k: 10) - 0.5) < 1e-12)
        // Two relevant, found at ranks 1 and 3: dcg = 1 + 0.5, ideal = 1 + 1/log2(3).
        let expected = 1.5 / (1 + 1 / log2(3.0))
        #expect(abs(Metrics.ndcg([true, false, true], relevantCount: 2, k: 10) - expected) < 1e-12)
        // Only one of two found at the top.
        #expect(abs(Metrics.ndcg([true], relevantCount: 2, k: 10) - 1 / (1 + 1 / log2(3.0))) < 1e-12)
        // Nothing relevant retrieved; nothing relevant exists.
        #expect(Metrics.ndcg([false, false], relevantCount: 1, k: 10) == 0)
        #expect(Metrics.ndcg([true], relevantCount: 0, k: 10) == 0)
        // Results below the cut-off do not count.
        #expect(Metrics.ndcg([false, true], relevantCount: 1, k: 1) == 0)
    }

    @Test func percentilesUseNearestRank() {
        let values = [10.0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        #expect(Metrics.percentile(values, 0.5) == 50)
        #expect(Metrics.percentile(values, 0.95) == 100)
        #expect(Metrics.percentile(values, 0) == 10)
        #expect(Metrics.percentile([], 0.5) == 0)
        #expect(Metrics.percentile([7], 0.95) == 7)
        #expect(Metrics.mean([1, 2, 3]) == 2 && Metrics.mean([]) == 0)
    }
}

@Suite("Relevance matching")
struct RelevanceTests {
    private func result(document: String, page: Int?, text: String) -> SearchResult {
        let doc = Document(
            id: DocumentID(rawValue: document), title: "title", uri: URL(fileURLWithPath: "/tmp/\(document)").absoluteString,
            kind: page == nil ? .text : .pdf, pageCount: nil, chunkCount: 1, contentHash: "h", addedAt: Date())
        return SearchResult(chunkID: ChunkID(rawValue: 1), document: doc, page: page, range: 0..<text.count, text: text, scores: .init())
    }

    @Test func matchesDocumentPageAndSnippet() {
        let hit = result(document: "a.pdf", page: 4, text: "The   trial enrolled\n412 participants.")
        #expect(Relevance.matches(hit, RelevantPassage(document: "a.pdf")))
        #expect(Relevance.matches(hit, RelevantPassage(document: "a.pdf", page: 4)))
        #expect(!Relevance.matches(hit, RelevantPassage(document: "a.pdf", page: 5)))
        #expect(!Relevance.matches(hit, RelevantPassage(document: "b.pdf")))
        #expect(Relevance.matches(hit, RelevantPassage(document: "a.pdf", snippet: "ENROLLED 412 participants")), "case and whitespace are ignored")
        #expect(!Relevance.matches(hit, RelevantPassage(document: "a.pdf", snippet: "412 enrolled")))
    }

    @Test func eachRelevantPassageIsCreditedOnce() {
        let first = result(document: "a.md", page: nil, text: "alpha fact")
        let second = result(document: "a.md", page: nil, text: "alpha fact again")
        let relevant = [RelevantPassage(document: "a.md", snippet: "alpha fact")]
        #expect(Relevance.flags(for: [first, second], relevant: relevant) == [true, false])
        let two = relevant + [RelevantPassage(document: "a.md", snippet: "again")]
        #expect(Relevance.flags(for: [first, second], relevant: two) == [true, true])
    }

    @Test func documentNameFallsBackToTitleWithoutAFileURI() {
        let doc = Document(
            id: DocumentID(rawValue: "x"), title: "Note", uri: nil, kind: .text, pageCount: nil, chunkCount: 1,
            contentHash: "h", addedAt: Date())
        #expect(Relevance.documentName(doc) == "Note")
    }
}

@Suite("Dataset format")
struct DatasetFormatTests {
    @Test func parsesQuestionsWithDefaultsAndSkipsBlankAndCommentLines() throws {
        let jsonl = """
        # a comment
        {"id":"a","question":"Q?","relevant":[{"document":"d.md","page":2,"snippet":"s"}],"answer":"A"}

        {"id":"b","question":"Unanswerable?","answerable":false}
        """
        let questions = try Dataset.parse(Data(jsonl.utf8))
        #expect(questions.count == 2)
        #expect(questions[0].answerable && questions[0].relevant == [RelevantPassage(document: "d.md", page: 2, snippet: "s")])
        #expect(!questions[1].answerable && questions[1].relevant.isEmpty)
    }

    @Test func rejectsMalformedAndInconsistentDatasets() throws {
        #expect(throws: EvalError.self) { try Dataset.parse(Data("{not json}".utf8)) }
        #expect(throws: EvalError.invalidDataset("there are no questions")) { try Dataset.validate([EvalQuestion](), documentsFolder: nil) }
        #expect(throws: EvalError.self) {
            try Dataset.validate([EvalQuestion(id: "a", question: "q"), EvalQuestion(id: "a", question: "q2", answerable: false)], documentsFolder: nil)
        }
        #expect(throws: EvalError.invalidDataset("answerable question “a” has no relevant passages")) {
            try Dataset.validate([EvalQuestion(id: "a", question: "q")], documentsFolder: nil)
        }
        #expect(throws: EvalError.self) {
            try Dataset.validate([EvalQuestion(id: "a", question: "q", relevant: [RelevantPassage(document: "d", page: 0)])], documentsFolder: nil)
        }
    }

    @Test func missingFilesAreReported() {
        let nowhere = FileManager.default.temporaryDirectory.appendingPathComponent("no-such-\(UUID().uuidString)")
        #expect(throws: EvalError.self) { try Dataset.folder(nowhere).load() }
        #expect(throws: EvalError.self) { try Dataset.jsonl(nowhere.appendingPathComponent("q.jsonl")).load() }
    }

    @Test func aReferenceToAMissingDocumentIsCaught() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("documents"), withIntermediateDirectories: true)
        try "x".write(to: directory.appendingPathComponent("documents/a.txt"), atomically: true, encoding: .utf8)
        try #"{"id":"q","question":"?","relevant":[{"document":"ghost.txt"}]}"#
            .write(to: directory.appendingPathComponent("questions.jsonl"), atomically: true, encoding: .utf8)
        #expect(throws: EvalError.self) { try Dataset.folder(directory).load() }
    }
}

@Suite("eval-small fixture dataset")
struct FixtureDatasetTests {
    @Test func loadsAndHasTheDocumentedShape() throws {
        let loaded = try Dataset.folder(evalSmall).load()
        #expect(loaded.name == "eval-small")
        #expect(loaded.questions.count == 49)
        #expect(loaded.questions.filter(\.answerable).count == 41)
        #expect(loaded.questions.filter { !$0.answerable }.count == 8)
    }

    /// Ground truth must be true: every snippet is in its document, and on its page for PDFs.
    @Test func everyRelevantSnippetOccursInItsDocumentAndPage() throws {
        let loaded = try Dataset.folder(evalSmall).load()
        let documents = try #require(loaded.documentsFolder)
        var parsed: [String: ParsedDocument] = [:]
        for name in try FileManager.default.contentsOfDirectory(atPath: documents.path) {
            parsed[name] = try DocumentParser.parse(fileAt: documents.appendingPathComponent(name))
        }
        #expect(parsed.count == 8)

        for question in loaded.questions {
            for passage in question.relevant {
                let document = try #require(parsed[passage.document], "\(question.id): missing \(passage.document)")
                guard let snippet = passage.snippet else { continue }
                let wanted = Relevance.squash(snippet)
                if let page = passage.page {
                    let ranges = try #require(document.pageRanges)
                    let range = try #require(ranges[page - 1], "\(question.id): page \(page) has no text")
                    #expect(Relevance.squash(document.text.scalarSlice(range)).contains(wanted), "\(question.id): “\(snippet)” not on page \(page)")
                } else {
                    #expect(Relevance.squash(document.text).contains(wanted), "\(question.id): “\(snippet)” not found")
                }
            }
        }
    }

    @Test func snippetsAreSpecificEnoughToIdentifyOnePassage() throws {
        let loaded = try Dataset.folder(evalSmall).load()
        let documents = try #require(loaded.documentsFolder)
        for question in loaded.questions {
            for passage in question.relevant {
                guard let snippet = passage.snippet, passage.page == nil else { continue }
                let document = try DocumentParser.parse(fileAt: documents.appendingPathComponent(passage.document))
                let text = Relevance.squash(document.text)
                let occurrences = text.components(separatedBy: Relevance.squash(snippet)).count - 1
                #expect(occurrences == 1, "\(question.id): snippet occurs \(occurrences) times")
            }
        }
    }
}

@Suite("Evaluator")
struct EvaluatorTests {
    private func withKnowledgeBase(_ body: (KnowledgeBase) async throws -> Void) async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let kb = try await KnowledgeBase(directory: directory.appendingPathComponent("kb"), services: await ModelServices(embedder: FakeEmbedder()))
        try await body(kb)
    }

    @Test func runsEveryModeOverAnswerableQuestionsOnly() async throws {
        try await withKnowledgeBase { kb in
            let report = try await Evaluator(knowledgeBase: kb).run(dataset: .folder(evalSmall))

            #expect(report.dataset.documents == 8 && report.dataset.questions == 49)
            #expect(report.dataset.answerable == 41 && report.dataset.unanswerable == 8)
            #expect(report.retrieval.map(\.mode) == ["keyword", "vector", "hybrid"])
            #expect(report.ingest.targetChunkTokens == 64 && report.ingest.overlapTokens == 8, "taken from config.json")
            #expect(report.ingest.chunks > 8)
            #expect(report.environment.embedder == "fake|64|0")

            for result in report.retrieval {
                let m = result.metrics
                #expect(m.questions == 41 && result.ranks.count == 41)
                #expect(m.recallAt1 <= m.recallAt5 && m.recallAt5 <= m.recallAt10)
                #expect(m.mrr >= m.recallAt1 * 1 - 1e-9 && m.mrr <= 1)
                #expect((0...1).contains(m.ndcgAt10))
                #expect(m.latencyP50Ms <= m.latencyP95Ms)
            }
            // Word-overlap retrieval on a corpus with distinct facts must do clearly better than chance.
            let keyword = try #require(report.result(for: "keyword"))
            #expect(keyword.metrics.recallAt5 > 0.7)
        }
    }

    @Test func theRerankModeIsAddedWhenARerankerIsConfiguredAndNotOtherwise() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let services = await ModelServices(embedder: FakeEmbedder(), reranker: FakeReranker(id: "fake-reranker|1|0"))
        let kb = try await KnowledgeBase(directory: directory.appendingPathComponent("kb"), services: services)
        let report = try await Evaluator(knowledgeBase: kb).run(dataset: .folder(evalSmall))
        #expect(report.retrieval.map(\.mode) == ["keyword", "vector", "hybrid", "hybrid+rerank"])
        #expect(report.environment.reranker == "fake-reranker|1|0")
        let rerank = try #require(report.result(for: "hybrid+rerank"))
        #expect(rerank.metrics.questions == 41 && rerank.metrics.recallAt5 > 0.7)
        #expect(report.markdown().contains("| hybrid+rerank |"))
    }

    @Test func askingForTheRerankModeWithoutARerankerIsAnError() async throws {
        try await withKnowledgeBase { kb in
            await #expect(throws: EvalError.rerankerNotConfigured) {
                try await Evaluator(knowledgeBase: kb, modes: [.hybridRerank]).run(dataset: .folder(evalSmall))
            }
        }
    }

    @Test func modeNamesRoundTrip() {
        for mode in [EvalMode.keyword, .vector, .hybrid, .hybridRerank] { #expect(EvalMode(name: mode.name) == mode) }
        #expect(EvalMode(name: "nonsense") == nil)
    }

    @Test func explicitIngestOptionsOverrideTheDatasetConfig() async throws {
        try await withKnowledgeBase { kb in
            let report = try await Evaluator(knowledgeBase: kb, modes: [.keyword], ingestOptions: IngestOptions(targetChunkTokens: 200, overlapTokens: 20))
                .run(dataset: .folder(evalSmall))
            #expect(report.ingest.targetChunkTokens == 200 && report.ingest.overlapTokens == 20)
        }
    }

    @Test func rankReportingMatchesTheRecallNumbers() async throws {
        try await withKnowledgeBase { kb in
            let report = try await Evaluator(kb: kb).run(dataset: .folder(evalSmall))
            for result in report.retrieval {
                let top5 = result.ranks.filter { ($0.rank ?? Int.max) <= 5 }.count
                #expect(abs(Double(top5) / 41 - result.metrics.recallAt5) < 1e-9)
            }
        }
    }

    @Test func aDocumentThatCannotBeAddedFailsTheRun() async throws {
        try await withKnowledgeBase { kb in
            let directory = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            try FileManager.default.createDirectory(at: directory.appendingPathComponent("documents"), withIntermediateDirectories: true)
            try "good content".write(to: directory.appendingPathComponent("documents/good.txt"), atomically: true, encoding: .utf8)
            try Data([1, 2, 3]).write(to: directory.appendingPathComponent("documents/bad.docx"))
            try #"{"id":"q","question":"content?","relevant":[{"document":"good.txt"}]}"#
                .write(to: directory.appendingPathComponent("questions.jsonl"), atomically: true, encoding: .utf8)
            await #expect(throws: EvalError.self) { try await Evaluator(knowledgeBase: kb).run(dataset: .folder(directory)) }
        }
    }

    @Test func jsonlDatasetsUseDocumentsAlreadyInTheKnowledgeBase() async throws {
        try await withKnowledgeBase { kb in
            for try await _ in kb.add(contentsOf: evalSmall.appendingPathComponent("documents")) {}
            let report = try await Evaluator(knowledgeBase: kb, modes: [.keyword])
                .run(dataset: .jsonl(evalSmall.appendingPathComponent("questions.jsonl")))
            #expect(report.retrieval.map(\.mode) == ["keyword"])
            #expect(report.retrieval[0].metrics.recallAt10 > 0.7)
        }
    }

    @Test func reportsRoundTripThroughJSONAndRenderMarkdown() async throws {
        try await withKnowledgeBase { kb in
            let report = try await Evaluator(knowledgeBase: kb, modes: [.keyword, .hybrid]).run(dataset: .folder(evalSmall))
            let directory = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let json = directory.appendingPathComponent("r.json"), markdown = directory.appendingPathComponent("r.md")
            try report.write(to: json, format: .json)
            try report.write(to: markdown, format: .markdown)
            #expect(try EvalReport.read(from: json).retrieval == report.retrieval)

            let text = try String(contentsOf: markdown, encoding: .utf8)
            #expect(text.contains("| keyword |") && text.contains("| hybrid |") && text.contains("Recall@5"))
            #expect(text.contains("eval-small") && text.contains("fake|64|0"))
        }
    }
}

private extension Evaluator {
    init(kb: KnowledgeBase) { self.init(knowledgeBase: kb) }
}

@Suite("Regression check")
struct RegressionTests {
    private func report(recallAt5: [String: Double]) -> EvalReport {
        EvalReport(
            dataset: DatasetSummary(name: "d", documents: 1, questions: 1, answerable: 1, unanswerable: 0),
            environment: EnvironmentSummary(device: "x", os: "y", embedder: "e", reranker: nil, generator: nil, generatedAt: Date(timeIntervalSince1970: 0)),
            ingest: IngestSummary(documents: 1, chunks: 1, seconds: 0, chunker: "structural", targetChunkTokens: 256, overlapTokens: 32, chunksPerSecond: 0),
            retrieval: recallAt5.sorted { $0.key < $1.key }.map { mode, value in
                ModeResult(
                    mode: mode,
                    metrics: RetrievalMetrics(
                        questions: 1, recallAt1: value, recallAt5: value, recallAt10: value, mrr: value, ndcgAt10: value,
                        latencyMeanMs: 0, latencyP50Ms: 0, latencyP95Ms: 0),
                    ranks: [])
            })
    }

    @Test func flagsDropsBeyondTheMarginOnly() {
        let baseline = report(recallAt5: ["hybrid": 0.90, "keyword": 0.80])
        #expect(report(recallAt5: ["hybrid": 0.90, "keyword": 0.80]).regressions(against: baseline).isEmpty)
        #expect(report(recallAt5: ["hybrid": 0.88, "keyword": 0.80]).regressions(against: baseline).isEmpty, "exactly the margin is allowed")
        #expect(report(recallAt5: ["hybrid": 0.95, "keyword": 0.80]).regressions(against: baseline).isEmpty, "improvements are fine")
        let regressions = report(recallAt5: ["hybrid": 0.85, "keyword": 0.80]).regressions(against: baseline)
        #expect(regressions.count == 1 && regressions[0].contains("hybrid"))
        #expect(report(recallAt5: ["hybrid": 0.85]).regressions(against: baseline, margin: 0.10).isEmpty)
    }

    @Test func modesMissingFromEitherReportAreIgnored() {
        let baseline = report(recallAt5: ["hybrid": 0.9, "vector": 0.9])
        #expect(report(recallAt5: ["hybrid": 0.9]).regressions(against: baseline).isEmpty)
    }
}
