import Foundation
internal import LociGrounding
internal import LociIngest
internal import LociLlama
internal import LociRetrieval
internal import LociStore
import os

private let log = Logger(subsystem: "dev.loci", category: "knowledge-base")

/// A local, searchable collection of documents.
///
/// Documents are parsed, chunked, embedded and stored in one SQLite file. Search combines
/// BM25 keyword search with vector search over the chunk embeddings. Reranking and answers
/// arrive in later milestones.
public actor KnowledgeBase {
    let store: Store
    let directory: URL
    private let services: ModelServices
    private let models: ModelSet
    private let options: KnowledgeBaseOptions
    private var isDestroyed = false

    /// Opens (or creates) a knowledge base stored under Application Support/Loci/<name> (the name is
    /// percent-encoded so it cannot escape that folder).
    ///
    /// Opening never loads a model. Operations that need the embedder throw
    /// `LociError.embedderMismatch` when the knowledge base was built with a different one; call
    /// `reindex()` to rebuild it for the configured embedder.
    public init(
        name: String,
        models: ModelSet = .recommended,
        options: KnowledgeBaseOptions = .default
    ) async throws {
        try await self.init(directory: try Self.defaultDirectory(for: name), models: models, options: options)
    }

    /// Opens a knowledge base at an explicit directory.
    public init(
        directory: URL,
        models: ModelSet = .recommended,
        options: KnowledgeBaseOptions = .default
    ) async throws {
        try await self.init(
            directory: directory, models: models, options: options,
            services: await ModelServices(models: models, options: options))
    }

    /// Opens a knowledge base with caller-supplied models (used by tests).
    init(directory: URL, models: ModelSet = .recommended, options: KnowledgeBaseOptions = .default, services: ModelServices) async throws {
        do {
            store = try Store(directory: directory, fileProtection: options.fileProtection)
        } catch {
            throw LociError.from(error)
        }
        self.directory = directory
        self.models = models
        self.options = options
        self.services = services
    }

    /// Application Support/Loci/<name>. The name is percent-encoded so it can never escape that folder.
    static func defaultDirectory(for name: String) throws -> URL {
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        let encoded = name.addingPercentEncoding(withAllowedCharacters: allowed) ?? "_"
        do {
            let support = try FileManager.default.url(
                for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            return support
                .appendingPathComponent("Loci", isDirectory: true)
                .appendingPathComponent(encoded.isEmpty ? "_" : encoded, isDirectory: true)
        } catch {
            throw LociError.storeCorrupted(reason: "Application Support is unavailable")
        }
    }

    // MARK: - Ingest

    /// Adds a document. Unchanged content already in the knowledge base is a no-op and returns the existing
    /// document. A file whose content changed replaces the earlier version stored for the same location.
    @discardableResult
    public func add(_ source: DocumentSource, options: IngestOptions = .default) async throws -> Document {
        try await ingest(source, options: options, report: { _ in }).document
    }

    /// Adds every supported file in `folder`, recursively. A file that cannot be added is reported as
    /// `.skipped` and the rest continue.
    public nonisolated func add(contentsOf folder: URL, options: IngestOptions = .default) -> AsyncThrowingStream<IngestProgress, Error> {
        stream { continuation in try await self.addFolder(folder, options: options, continuation: continuation) }
    }

    /// Re-embeds every passage with the configured embedder and records it as the knowledge base's
    /// embedder. Use it after changing `ModelSet.embedder`. Searches that need vectors fail with
    /// `embedderMismatch` until it finishes; cancelling is safe and it can be run again.
    public nonisolated func reindex() -> AsyncThrowingStream<IngestProgress, Error> {
        stream { continuation in try await self.runReindex(continuation: continuation) }
    }

    private nonisolated func stream<Element: Sendable>(
        _ body: @escaping @Sendable (AsyncThrowingStream<Element, Error>.Continuation) async throws -> Void
    ) -> AsyncThrowingStream<Element, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await body(continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: LociError.from(error))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Removes a document and all its passages. Removing an unknown ID does nothing.
    public func remove(_ id: DocumentID) async throws {
        try ensureOpen()
        do {
            try await store.remove(documentID: id.rawValue)
        } catch {
            throw LociError.from(error)
        }
    }

    /// All documents, oldest first.
    public func documents() async throws -> [Document] {
        try ensureOpen()
        do {
            return try await store.documents().map(Document.init)
        } catch {
            throw LociError.from(error)
        }
    }

    /// The normalised text of a document: what every `Citation.range` and `SearchResult.range` refers to, so
    /// `text.unicodeScalars[range]` is exactly the excerpt. Nil if the document is not in the knowledge base.
    public func documentText(for id: DocumentID) async throws -> String? {
        try ensureOpen()
        do {
            return try await store.documentText(id: id.rawValue)
        } catch {
            throw LociError.from(error)
        }
    }

    /// For a PDF, the part of `documentText` that came from each page (index 0 is page 1); nil for a page with no
    /// text, and nil overall for other formats. A PDF citation's `range` always lies inside its page's range.
    public func pageRanges(for id: DocumentID) async throws -> [Range<Int>?]? {
        try ensureOpen()
        do {
            return try await store.pageRanges(documentID: id.rawValue)
        } catch {
            throw LociError.from(error)
        }
    }

    // MARK: - Retrieve

    /// Searches the knowledge base.
    ///
    /// `.keyword` needs no model. `.vector` and `.hybrid` embed the query with the configured embedder
    /// and throw `embedderMismatch` if the knowledge base was built with another one. Results are
    /// passages: overlapping or adjacent chunks of one document are merged.
    public func search(_ query: String, options: RetrievalOptions = .default) async throws -> [SearchResult] {
        try await retrieve(query, options: options).results
    }

    /// Retrieved passages with the time spent retrieving and reranking them.
    struct Retrieval {
        var results: [SearchResult]
        var retrievalTime: Duration
        var rerankTime: Duration
    }

    func retrieve(_ query: String, options: RetrievalOptions) async throws -> Retrieval {
        try ensureOpen()
        guard options.limit > 0, !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return Retrieval(results: [], retrievalTime: .zero, rerankTime: .zero)
        }
        do {
            var allowed: Set<String>?
            switch options.filter {
            case .none:
                break
            case .document(let ids):
                allowed = Set(ids.map(\.rawValue))
            case .metadata(let key, let value):
                allowed = Set(try await store.documents().filter { $0.metadata[key] == value }.map(\.id))
            }

            let start = ContinuousClock.now
            var queryVector: [Float]?
            if options.mode != .keyword {
                try await ensureEmbedderCompatible()
                queryVector = try await services.withEmbedder { try await $0.embed([query], role: .query).first }
            }
            let rerankTimer = Stopwatch()
            var reranking: Reranking?
            if options.rerank, await services.hasReranker {
                let services = self.services
                reranking = Reranking(depth: Self.rerankDepth) { query, passages in
                    try await rerankTimer.measure { try await services.rerank(query: query, passages: passages) }
                }
            }
            let strategy: RetrievalStrategy = switch options.mode {
            case .keyword: .keyword
            case .vector: .vector
            case .hybrid: .hybrid
            }
            let passages = try await RetrievalPipeline(store: store).retrieve(RetrievalRequest(
                query: query, queryVector: queryVector, limit: options.limit, candidatePool: options.candidatePool,
                strategy: strategy, documentIDs: allowed, reranking: reranking))

            var documents: [String: Document] = [:]
            var results: [SearchResult] = []
            for passage in passages {
                if documents[passage.documentID] == nil, let stored = try await store.document(id: passage.documentID) {
                    documents[passage.documentID] = Document(stored)
                }
                guard let document = documents[passage.documentID] else { continue }
                results.append(SearchResult(
                    chunkID: ChunkID(rawValue: passage.chunkID), document: document, page: passage.page,
                    range: passage.range, text: passage.text,
                    scores: .init(bm25: passage.bm25, vector: passage.vector, fused: passage.fused, rerank: passage.rerank)))
            }
            let total = ContinuousClock.now - start
            let rerankTime = rerankTimer.total
            log.info("search returned \(results.count) results in \(total)")
            return Retrieval(results: results, retrievalTime: total - rerankTime, rerankTime: rerankTime)
        } catch {
            throw LociError.from(error)
        }
    }

    /// The version of the grounding prompt answers are generated with. Evaluation results are only comparable between
    /// answers made with the same version.
    public static var groundingPromptVersion: Int { PromptVersion.current.rawValue }

    // MARK: - Support scoring

    /// How well each passage supports `claim`, from 0 (not at all) to 1: the score verification gives a sentence of an answer.
    ///
    /// With a reranker this is the probability the cross-encoder assigns to the passage supporting the claim;
    /// without one it is the cosine similarity of their embeddings. The two scales differ, which is why
    /// `AnswerOptions.supportThresholds` are calibrated per model set (see `docs/MODELS.md`).
    public func supportScores(for claim: String, against passages: [String]) async throws -> [Float] {
        try ensureOpen()
        do {
            return try await supportScorer()(claim, passages)
        } catch {
            throw LociError.from(error)
        }
    }

    private func supportScorer() async throws -> SupportScorer {
        let services = self.services
        if await services.hasReranker {
            return { claim, passages in
                try await services.rerank(query: claim, passages: passages, instruction: GroundingPrompt.verificationInstruction)
                    .map(Self.probability)
            }
        }
        try await ensureEmbedderCompatible()
        return { claim, passages in
            try await services.withEmbedder { embedder in
                let claimVector = try await embedder.embed([claim], role: .query)
                let passageVectors = try await embedder.embed(passages, role: .document)
                return passageVectors.map { vector in
                    let dot = zip(claimVector[0], vector).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                    return min(max(dot, 0), 1)
                }
            }
        }
    }

    /// The logistic function: turns a reranker's raw score into a 0–1 probability.
    static func probability(_ score: Float) -> Float {
        1 / (1 + exp(-max(min(score, 30), -30)))
    }

    // MARK: - Answer

    /// Answers `question` from the documents, with `[n]` citations.
    ///
    /// Retrieves passages, fits the best of them into the generator's context, and asks the generator to answer
    /// using only those sources. Throws `LociError.noGenerator` when the `ModelSet` has no generator.
    /// With `AnswerOptions.verify` (the default) every sentence is checked against the sources it cites: see `AnswerEvent.sentence`
    /// and `Answer.support`.
    public func ask(_ question: String, options: AnswerOptions = .default) async throws -> Answer {
        var answer: Answer?
        for try await event in streamAnswer(question, options: options) {
            if case .finished(let finished) = event { answer = finished }
        }
        guard let answer else { throw LociError.cancelled }
        return answer
    }

    /// Like `ask`, reporting progress: the retrieved sources first, then the reply as it is generated, then the finished answer.
    public nonisolated func streamAnswer(_ question: String, options: AnswerOptions = .default) -> AsyncThrowingStream<AnswerEvent, Error> {
        stream { continuation in try await self.runAnswer(question, options: options, continuation: continuation) }
    }

    private struct Generated: Sendable {
        var text: String
        var stats: GenerationStats?
        var sourceCount: Int
        var promptTokens: Int
    }

    private func runAnswer(
        _ question: String, options: AnswerOptions, continuation: AsyncThrowingStream<AnswerEvent, Error>.Continuation
    ) async throws {
        try ensureOpen()
        guard await services.hasGenerator else { throw LociError.noGenerator }

        let retrieval = try await retrieve(question, options: options.retrieval)
        continuation.yield(.retrieved(retrieval.results))

        func stats(
            timeToFirstToken: Duration = .zero, tokensPerSecond: Double = 0, contextTokens: Int = 0, verificationTime: Duration = .zero
        ) -> AnswerStats {
            AnswerStats(
                retrievalTime: retrieval.retrievalTime, rerankTime: retrieval.rerankTime, timeToFirstToken: timeToFirstToken,
                generationTokensPerSecond: tokensPerSecond, verificationTime: verificationTime, contextTokens: contextTokens,
                peakMemoryBytes: ProcessMemory.peakFootprintBytes)
        }

        // Nothing relevant was found: say so without spending time on generation.
        guard !retrieval.results.isEmpty else {
            continuation.yield(.finished(Self.refusal(sources: [], stats: stats())))
            return
        }

        let sources = retrieval.results.map { PromptSource(title: $0.document.title, page: $0.page, text: $0.text) }
        let generated: Generated
        do {
            generated = try await services.withGenerator { generator in
                let budget = options.maxContextTokens ?? max(generator.contextTokens - options.maxAnswerTokens, 0)
                guard let packed = try await ContextPacker.pack(
                    question: question, sources: sources, instructions: options.instructions, maxTokens: budget,
                    sanitize: generator.sanitize, countTokens: generator.promptTokenCount)
                else {
                    throw LociError.backendFailure(operation: "Fitting the best source into the model's context", code: -1)
                }

                var text = ""
                var finalStats: GenerationStats?
                for try await event in generator.generate(
                    packed.messages, parameters: GenerationParameters(maxTokens: options.maxAnswerTokens, temperature: options.temperature)
                ) {
                    switch event {
                    case .text(let piece):
                        text += piece
                        continuation.yield(.token(piece))
                    case .finished(let result):
                        finalStats = result
                    }
                }
                return Generated(text: text, stats: finalStats, sourceCount: packed.sourceCount, promptTokens: packed.promptTokens)
            }
        } catch {
            throw LociError.from(error)
        }

        let included = Array(retrieval.results.prefix(generated.sourceCount))
        let timeToFirstToken = generated.stats?.timeToFirstToken ?? .zero
        let tokensPerSecond = generated.stats?.tokensPerSecond ?? 0
        let rawText = generated.text.trimmingCharacters(in: .whitespacesAndNewlines)

        // The model declined, or said nothing: there is nothing to verify.
        guard options.verify, !GroundingPrompt.isRefusal(rawText) else {
            let answer = Self.assemble(
                rawText: rawText, sources: included,
                stats: stats(timeToFirstToken: timeToFirstToken, tokensPerSecond: tokensPerSecond, contextTokens: generated.promptTokens))
            continuation.yield(.finished(answer))
            return
        }

        let verificationStart = ContinuousClock.now
        let verification: VerificationResult
        do {
            verification = try await AnswerVerifier.verify(
                rawText, sources: included.map(\.text),
                thresholds: SupportThresholds(supported: options.supportThresholds.supported, weak: options.supportThresholds.weak),
                scorer: try await supportScorer(),
                onSentence: { sentence in
                    continuation.yield(.sentence(Self.answeredSentence(sentence)))
                })
        } catch {
            throw LociError.from(error)
        }
        let verificationTime = ContinuousClock.now - verificationStart
        let recovered = verification.sentences.filter(\.recovered).count
        log.info("verified \(verification.sentences.count) sentences, \(recovered) citations recovered, in \(verificationTime)")

        let finalStats = stats(
            timeToFirstToken: timeToFirstToken, tokensPerSecond: tokensPerSecond, contextTokens: generated.promptTokens,
            verificationTime: verificationTime)

        // Most of the answer is not backed by its sources: say so instead of presenting it.
        if options.refuseWhenUnsupported, verification.mostlyUnsupported {
            continuation.yield(.finished(Self.refusal(sources: included, stats: finalStats)))
            return
        }

        let summary = SupportSummary(
            supportedFraction: verification.supportedFraction, level: SupportLevel(verification.overall))
        continuation.yield(.finished(Answer(
            text: verification.text, sentences: verification.sentences.map(Self.answeredSentence),
            citations: Self.citations(usedSources: verification.usedSources, sources: included),
            support: summary, refused: false, stats: finalStats)))
    }

    private static func answeredSentence(_ sentence: VerifiedSentence) -> AnsweredSentence {
        AnsweredSentence(
            text: sentence.text, citationIndices: sentence.citationIndices, support: SupportLevel(sentence.label), score: sentence.score)
    }

    /// The numbered citations for `usedSources` (the model's source numbers in order of first use).
    private static func citations(usedSources: [Int], sources: [SearchResult]) -> [Citation] {
        usedSources.enumerated().compactMap { offset, original in
            guard sources.indices.contains(original - 1) else { return nil }
            return citation(for: sources[original - 1], index: offset + 1)
        }
    }

    private static func citation(for source: SearchResult, index: Int) -> Citation {
        Citation(
            index: index, document: source.document, chunkID: source.chunkID, page: source.page, range: source.range,
            excerpt: source.text,
            retrievalScore: source.scores.rerank ?? source.scores.fused ?? source.scores.vector ?? source.scores.bm25 ?? 0)
    }

    /// A refusal. `citations` lists the passages that were looked at, in rank order, so the user can inspect them.
    private static func refusal(sources: [SearchResult], stats: AnswerStats) -> Answer {
        Answer(
            text: GroundingPrompt.refusal, sentences: [],
            citations: sources.enumerated().map { citation(for: $1, index: $0 + 1) },
            support: nil, refused: true, stats: stats)
    }

    /// Builds an unverified answer from the model's raw reply: resolves citation markers and detects a refusal.
    static func assemble(rawText: String, sources: [SearchResult], stats: AnswerStats) -> Answer {
        let trimmed = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        if GroundingPrompt.isRefusal(trimmed) { return refusal(sources: sources, stats: stats) }
        let parsed = CitationParser.parse(trimmed, sourceCount: sources.count)
        return Answer(
            text: parsed.text, sentences: [], citations: citations(usedSources: parsed.usedSources, sources: sources),
            support: nil, refused: false, stats: stats)
    }

    /// How many of the best fused candidates the reranker rescores.
    static let rerankDepth = 20

    // MARK: - Lifecycle

    /// Counts, embedder and size on disk. Empty once the knowledge base is destroyed.
    public var stats: KnowledgeBaseStats {
        get async {
            guard !isDestroyed, let stats = try? await store.stats() else {
                return KnowledgeBaseStats(
                    documentCount: 0, chunkCount: 0, vectorCount: 0, databaseBytes: 0, embedderID: nil, rerankerID: nil, generatorID: nil)
            }
            let stored = try? await store.storedEmbedder()
            let reranker = try? await services.rerankerID()
            let generator = try? await services.generatorID()
            return KnowledgeBaseStats(
                documentCount: stats.documentCount, chunkCount: stats.chunkCount, vectorCount: stats.vectorCount,
                databaseBytes: stats.databaseBytes, embedderID: stored?.id, rerankerID: reranker, generatorID: generator)
        }
    }

    /// Unloads resident models; they load again on next use.
    public func unloadModels() async {
        await services.unloadAll()
    }

    /// Deletes every file for this knowledge base. The instance is unusable afterwards.
    public func destroy() async throws {
        guard !isDestroyed else { return }
        isDestroyed = true
        await services.unloadAll()
        await store.close()
        do {
            if FileManager.default.fileExists(atPath: directory.path) {
                try FileManager.default.removeItem(at: directory)
            }
        } catch {
            throw LociError.storeCorrupted(reason: "the knowledge base files could not be deleted")
        }
    }

    // MARK: - Internals

    private func ensureOpen() throws {
        if isDestroyed { throw LociError.storeCorrupted(reason: "the knowledge base was destroyed") }
    }

    /// Throws `embedderMismatch` if the stored vectors come from a different embedder than the configured one.
    /// An empty knowledge base simply adopts the configured embedder.
    private func ensureEmbedderCompatible() async throws {
        let configured = try await services.embedderIdentity()
        let stored = try await store.storedEmbedder()
        if stored?.id == configured.id { return }
        if try await store.stats().chunkCount == 0 {
            try await store.setEmbedder(StoreEmbedder(id: configured.id, dimension: configured.dimension))
            return
        }
        throw LociError.embedderMismatch(stored: stored?.id ?? "none", configured: configured.id)
    }

    private func addFolder(
        _ folder: URL, options: IngestOptions,
        continuation: AsyncThrowingStream<IngestProgress, Error>.Continuation
    ) async throws {
        try ensureOpen()
        let files = Self.files(in: folder)
        for (index, url) in files.enumerated() {
            try Task.checkCancellation()
            let name = url.lastPathComponent
            @Sendable func send(_ phase: IngestProgress.Phase, completed: Int) {
                continuation.yield(IngestProgress(document: name, phase: phase, completed: completed, total: files.count))
            }
            do {
                let outcome = try await ingest(.file(url), options: options, report: { send($0, completed: index) })
                send(outcome.skipped ? .skipped("unchanged") : .done, completed: index + 1)
            } catch {
                let failure = LociError.from(error, url: url)
                switch failure {
                case .cancelled, .embedderMismatch, .modelNotFound, .invalidModel, .modelRoleMismatch:
                    throw failure  // the same problem would hit every file
                default:
                    send(.skipped(failure.errorDescription ?? "failed"), completed: index + 1)
                }
            }
        }
    }

    private static func files(in folder: URL) -> [URL] {
        let keys: [URLResourceKey] = [.isRegularFileKey]
        guard let walker = FileManager.default.enumerator(
            at: folder, includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { return [] }
        var urls: [URL] = []
        for case let url as URL in walker where (try? url.resourceValues(forKeys: Set(keys)))?.isRegularFile == true {
            urls.append(url)
        }
        return urls.sorted { $0.path < $1.path }
    }

    private func ingest(
        _ source: DocumentSource, options: IngestOptions,
        report: @Sendable (IngestProgress.Phase) -> Void
    ) async throws -> (document: Document, skipped: Bool) {
        try ensureOpen()
        var location: URL?
        do {
            if case .file(let url) = source { location = url }
            try Task.checkCancellation()
            report(.parsing)
            let parsed = try await Self.parse(source)

            // Unchanged content: no chunking and, above all, no embedding.
            if let existing = try await store.document(hash: parsed.document.contentHash) {
                return (Document(existing), true)
            }
            try await ensureEmbedderCompatible()
            try Task.checkCancellation()

            // The embedder stays loaded for chunking (its tokenizer sizes the chunks) and embedding.
            let parsedDocument = parsed.parsed
            let (chunks, vectors, identity) = try await services.withEmbedder { embedder in
                report(.chunking)
                let chunks = await Self.chunk(parsedDocument, options: options, counter: EmbedderTokenCounter(embedder: embedder))
                try Task.checkCancellation()
                report(.embedding)
                return (chunks, try await Self.embed(chunks.map(\.text), with: embedder), embedder.identity)
            }
            try Task.checkCancellation()

            report(.storing)
            var document = parsed.document
            document.metadata = options.metadata
            let outcome = try await store.insert(
                document,
                chunks: chunks.map {
                    NewChunk(
                        ordinal: $0.ordinal, page: $0.page, start: $0.range.lowerBound, end: $0.range.upperBound,
                        headingPath: $0.headingPath, tokenCount: $0.tokenCount, text: $0.text)
                },
                vectors: vectors, embedder: StoreEmbedder(id: identity.id, dimension: identity.dimension))
            switch outcome {
            case .inserted(let stored): return (Document(stored), false)
            case .existing(let stored): return (Document(stored), true)
            }
        } catch {
            throw LociError.from(error, url: location)
        }
    }

    private struct Parsed: Sendable {
        let document: NewDocument
        let parsed: ParsedDocument
    }

    /// Parses off the actor so a large document does not block other calls.
    private static func parse(_ source: DocumentSource) async throws -> Parsed {
        let parsed: ParsedDocument
        let title: String
        let uri: String?
        switch source {
        case .file(let url):
            parsed = try DocumentParser.parse(fileAt: url)
            title = parsed.title ?? url.deletingPathExtension().lastPathComponent
            uri = url.standardizedFileURL.absoluteString
        case .text(let text, let given, let givenURI):
            parsed = DocumentParser.parse(text: text, kind: .text)
            guard !parsed.text.isEmpty else {
                throw IngestError.unreadable(reason: "the document contains no text")
            }
            title = given
            uri = givenURI
        }
        let document = NewDocument(
            title: title, uri: uri, kind: parsed.kind.rawValue, pageCount: parsed.pageCount,
            contentHash: ContentHash.sha256(parsed.text), metadata: [:], text: parsed.text,
            pageRanges: parsed.pageRanges)
        return Parsed(document: document, parsed: parsed)
    }

    private static func chunk(_ parsed: ParsedDocument, options: IngestOptions, counter: EmbedderTokenCounter) async -> [LociIngest.Chunk] {
        Chunker.chunk(
            parsed,
            options: ChunkingOptions(
                kind: LociIngest.ChunkerKind(rawValue: options.chunker.rawValue) ?? .structural,
                targetTokens: options.targetChunkTokens, overlapTokens: options.overlapTokens),
            counter: counter)
    }

    /// Embeds in slices so cancellation is honoured between them.
    private static func embed(_ texts: [String], with embedder: any TextEmbedder) async throws -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        for start in stride(from: 0, to: texts.count, by: 32) {
            try Task.checkCancellation()
            vectors += try await embedder.embed(Array(texts[start..<min(start + 32, texts.count)]), role: .document)
        }
        return vectors
    }

    private func runReindex(continuation: AsyncThrowingStream<IngestProgress, Error>.Continuation) async throws {
        try ensureOpen()
        let identity = try await services.embedderIdentity()
        let target = StoreEmbedder(id: identity.id, dimension: identity.dimension)
        let documents = try await store.documents()

        for (index, document) in documents.enumerated() {
            try Task.checkCancellation()
            @Sendable func send(_ phase: IngestProgress.Phase, completed: Int) {
                continuation.yield(IngestProgress(document: document.title, phase: phase, completed: completed, total: documents.count))
            }
            send(.embedding, completed: index)
            let chunks = try await store.chunks(documentID: document.id)
            let texts = chunks.map(\.text)
            let vectors = try await services.withEmbedder { try await Self.embed(texts, with: $0) }
            try await store.replaceVectors(documentID: document.id, vectors: vectors, embedder: target)
            send(.done, completed: index + 1)
        }
        try await store.setEmbedder(target)
    }
}

extension Document {
    init(_ stored: StoredDocument) {
        self.init(
            id: DocumentID(rawValue: stored.id), title: stored.title, uri: stored.uri,
            kind: DocumentKind(rawValue: stored.kind) ?? .text, pageCount: stored.pageCount,
            chunkCount: stored.chunkCount, contentHash: stored.contentHash, addedAt: stored.addedAt)
    }
}


/// Accumulates the time spent in the calls it measures.
final class Stopwatch: Sendable {
    private let elapsed = OSAllocatedUnfairLock(initialState: Duration.zero)

    var total: Duration { elapsed.withLock { $0 } }

    func measure<T: Sendable>(_ body: () async throws -> T) async rethrows -> T {
        let start = ContinuousClock.now
        defer {
            let spent = ContinuousClock.now - start
            elapsed.withLock { $0 += spent }
        }
        return try await body()
    }
}

/// The app's own memory use, as reported by the kernel. Used by benchmarks and diagnostics.
public enum MemoryUsage {
    /// Bytes the process is currently charged for.
    public static var currentBytes: UInt64 { ProcessMemory.footprintBytes }

    /// The highest footprint the process has reached so far.
    public static var peakBytes: UInt64 { ProcessMemory.peakFootprintBytes }
}
