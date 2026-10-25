# Loci — Technical Specification

> **Status:** Draft v0.2 · **Owner:** Melvin Ramkhelawan · **Last updated:** 2026-10-24
>
> Name check before the first public tag: GitHub, Swift Package Index, trademark search.

---

## 1. Purpose

Loci is a Swift package for **grounded, cited, on-device question answering over a user's documents** on Apple platforms. It runs the entire retrieval-augmented generation (RAG) loop in-process on llama.cpp with Metal:

```
documents → parse → chunk → embed → hybrid retrieve → rerank → generate → cite → verify
```

The name comes from the *method of loci*, the memory-palace technique: knowledge is stored in places and recalled by walking back to them. Loci stores documents locally and recalls the exact place each answer came from.

### 1.1 The promise

```swift
let kb = try await KnowledgeBase(name: "papers")
try await kb.add(contentsOf: papersFolder)

let answer = try await kb.ask("What sample size did the 2024 trial use?")
print(answer.text)          // "The trial enrolled 412 participants [1]."
print(answer.citations[0])  // trial.pdf · page 4 · "…412 participants were randomised…"
```

### 1.2 What makes Loci different

See `docs/POSITIONING.md` for the full comparison. In short, existing Swift libraries each cover one segment (storage, ingestion, embeddings or generation). Loci is the only one that:

1. **Closes the loop** — documents in, cited answer out, in one package.
2. **Verifies citations** — every sentence is linked to source passages and scored for support; unsupported claims are flagged.
3. **Runs one native GGUF stack** — embedder, reranker and generator all in-process via llama.cpp + Metal, under one memory budget. No servers.
4. **Measures itself** — a built-in evaluation harness reports retrieval recall, citation accuracy, faithfulness, latency and memory per device.
5. **Treats data as sensitive by default** — no network in the core, iOS Data Protection on every file, nothing logged in plaintext.

### 1.3 Goals

| # | Goal | Measure |
|---|------|---------|
| G1 | Documents to cited answer in ≤ 3 lines | README hero example |
| G2 | Every answer is traceable | 100% of answer sentences carry a citation or an explicit `.unsupported` flag |
| G3 | Measured quality | `LociEval` produces recall@k, citation precision/recall and faithfulness on any dataset |
| G4 | Fits on a phone | Full pipeline runs on a 6 GB iPhone with the recommended model set |
| G5 | Private | Core targets make zero network calls (enforced in CI) |
| G6 | Swift-native | async/await, `AsyncSequence` streaming, `Sendable`, Swift 6 strict concurrency, typed errors |
| G7 | Stable despite upstream churn | llama.cpp pinned; never visible in public API |

### 1.4 Non-goals (v1)

- Cloud LLMs or cloud embeddings (users can still plug them in via protocols later; Loci won't ship them).
- Images, audio, video retrieval (text only in v1; scanned PDFs via Vision OCR is v1.x).
- Agent memory / tool frameworks (Wax covers this space).
- Training or fine-tuning.
- Linux.

---

## 2. Platforms & toolchain

| Item | Minimum | Notes |
|------|---------|-------|
| Swift tools | 6.0 | Swift 6 language mode, strict concurrency |
| Xcode | 16 | |
| iOS / iPadOS | 16.4 | Physical devices only: the pinned llama.cpp binary has no Simulator slice. Floor set by the same binary. |
| macOS | 13.3 | Apple Silicon recommended. Floor set by the pinned llama.cpp binary. |

iOS 16 is a deliberate advantage: the closest alternatives require iOS 18+.

---

## 3. Package layout

```
Loci/
├── Package.swift
├── Sources/
│   ├── Loci/                    # PUBLIC API — what apps import
│   │   ├── KnowledgeBase.swift
│   │   ├── Answer.swift          # Answer, Citation, SupportLevel, AnswerEvent
│   │   ├── SearchResult.swift
│   │   ├── Document.swift        # Document, DocumentID, DocumentSource
│   │   ├── ModelSet.swift        # embedder / reranker / generator choice
│   │   ├── Options/              # IngestOptions, RetrievalOptions, AnswerOptions
│   │   ├── LociError.swift
│   │   └── Loci.docc/
│   ├── LociIngest/              # Parsers + chunkers (internal)
│   │   ├── Parsers/              # PDF (PDFKit), Markdown, HTML, PlainText
│   │   └── Chunkers/             # Structural, Sentence, TokenWindow
│   ├── LociStore/               # SQLite store: chunks, FTS5, vectors (internal)
│   ├── LociRetrieval/           # Hybrid search, RRF fusion, rerank, context packing (internal)
│   ├── LociGrounding/           # Prompt building, citation parsing, support scoring (internal)
│   ├── LociLlama/               # The ONLY target that imports llama.cpp (internal)
│   │   ├── LlamaBackend.swift
│   │   ├── LlamaModel.swift
│   │   ├── LlamaContext.swift
│   │   ├── Embedder.swift        # pooling: mean/cls/last per model metadata
│   │   ├── Reranker.swift        # rank pooling
│   │   ├── Generator.swift       # chat template, sampler, streaming
│   │   └── ModelResidency.swift  # memory budget + load/unload policy
│   ├── LociEval/                # PUBLIC, optional: evaluation harness
│   └── LociHub/                 # PUBLIC, optional: model downloads (the only target allowed network)
├── Tests/
│   ├── LociTests/               # unit, no models
│   ├── LociIntegrationTests/    # tiny GGUF fixtures, CPU only
│   └── Fixtures/                # sample PDFs/MD/HTML + eval JSONL
├── Examples/
│   ├── LociReader/              # SwiftUI app: import files, ask, tap citation → jump to page
│   └── loci-cli/                # index, ask, search, eval, bench
├── Scripts/
│   ├── update-llama.sh
│   └── fetch-test-models.sh
└── docs/
```

### 3.1 Products

| Product | Includes | Network |
|---------|----------|---------|
| `Loci` | Loci + internal targets | **None** (CI-enforced) |
| `LociEval` | Evaluation harness | None |
| `LociHub` | Hugging Face download, checksum, cache | Yes — the only one |
| `loci-cli` | Executable | Via LociHub only |

Splitting internals into targets keeps compile times down and makes the layering enforceable; only `Loci`, `LociEval` and `LociHub` are products.

### 3.2 Dependencies

| Dependency | How | Why |
|------------|-----|-----|
| llama.cpp | Official release XCFramework as `binaryTarget` (URL + SHA-256), pinned | Embedding, reranking, generation with Metal |
| SQLite | System `libsqlite3` **if FTS5 is available on all target OS versions**; otherwise vendored amalgamation as a C target | Chunk store + BM25 full-text search |
| PDFKit, NaturalLanguage, Accelerate, CryptoKit, os | Apple SDK | Parsing, sentence splitting, vector maths, hashing, logging |

**No third-party Swift dependencies** in `Loci`.

> **M0 verification task:** confirm FTS5 is compiled into the system SQLite on iOS 16 and macOS 13 (`PRAGMA compile_options;`). If not, vendor the amalgamation with `SQLITE_ENABLE_FTS5`.
>
> **M0 verification task:** confirm the exact llama.cpp XCFramework asset name on the current release page, and that the pinned release supports embedding pooling and rank pooling for rerankers.

> **M0 results (2026-10-24):**
> - llama.cpp pinned to `b11849`; asset is `llama-b11849-xcframework.zip` (SHA-256 recorded in `Package.swift`). It ships **macOS and iOS-device slices only** — no iOS Simulator, visionOS or Catalyst slice, so those platforms are out of scope.
> - `llama.h` exposes `LLAMA_POOLING_TYPE_{MEAN,CLS,LAST,RANK}` and `llama_get_embeddings_seq`; an embedding round-trip is covered by `LlamaSupportTests`.
> - FTS5 is present in the system SQLite on macOS 26 (`SQLiteCapabilitiesTests`). **Not yet verified on iOS 16 / macOS 13**; no such runtime is available locally. Until it is, treat the system-SQLite decision as provisional.

### 3.3 Storage decision

Loci owns its storage rather than building on VecturaKit or Wax because:
- citations need exact character offsets and page numbers per chunk, which must be first-class columns;
- verification needs the original chunk text co-located with vectors;
- VecturaKit requires iOS 18+;
- one SQLite file is easy to protect with iOS Data Protection and to delete atomically.

A `VecturaKit` adapter may be offered later as an optional integration.

---

## 4. Public API

All public types are `Sendable`. No llama.cpp or SQLite type appears in public API.

### 4.1 KnowledgeBase

```swift
public actor KnowledgeBase {
    /// Opens (or creates) a knowledge base stored under Application Support/Loci/<name> (the name is percent-encoded so it cannot escape that folder).
    public init(
        name: String,
        models: ModelSet = .recommended,
        options: KnowledgeBaseOptions = .default
    ) async throws

    /// Opens a knowledge base at an explicit directory.
    public init(directory: URL, models: ModelSet = .recommended, options: KnowledgeBaseOptions = .default) async throws

    // MARK: Ingest
    @discardableResult
    public func add(_ source: DocumentSource, options: IngestOptions = .default) async throws -> Document
    public func add(contentsOf folder: URL, options: IngestOptions = .default) -> AsyncThrowingStream<IngestProgress, Error>
    public func remove(_ id: DocumentID) async throws
    public func documents() async throws -> [Document]
    public func reindex() -> AsyncThrowingStream<IngestProgress, Error>   // after model change

    // MARK: Retrieve
    public func search(_ query: String, options: RetrievalOptions = .default) async throws -> [SearchResult]

    // MARK: Answer
    public func ask(_ question: String, options: AnswerOptions = .default) async throws -> Answer
    public func streamAnswer(_ question: String, options: AnswerOptions = .default) -> AsyncThrowingStream<AnswerEvent, Error>

    // MARK: Lifecycle
    public var stats: KnowledgeBaseStats { get async }          // zeros after destroy()
    public func unloadModels() async
    public func destroy() async throws      // deletes all files for this knowledge base
}
```

### 4.2 Documents

```swift
public enum DocumentSource: Sendable {
    case file(URL)                                   // pdf, md, markdown, html, htm, txt
    case text(String, title: String, uri: String? = nil)
}

public struct Document: Sendable, Identifiable, Hashable, Codable {
    public let id: DocumentID
    public let title: String
    public let uri: String?
    public let kind: DocumentKind        // .pdf, .markdown, .html, .text
    public let pageCount: Int?
    public let chunkCount: Int
    public let contentHash: String       // SHA-256 of normalised text — re-adding unchanged docs is a no-op
    public let addedAt: Date
}

public struct IngestProgress: Sendable {
    public let document: String          // file name or title
    public let phase: Phase              // .parsing, .chunking, .embedding, .storing, .done, .skipped(reason)
    public let completed: Int            // documents finished in this operation
    public let total: Int
}
```

`DocumentID` and `ChunkID` are opaque `Hashable, Sendable, Codable` wrappers (a UUID string and a row ID).

Ingest semantics:
- Re-adding a document whose normalised text is unchanged returns the existing `Document` and writes nothing (`.skipped("unchanged")` in folder progress).
- A file whose content changed replaces the earlier version stored for the same `uri`.
- `add(contentsOf:)` walks the folder recursively, ignores hidden files, and reports a file that cannot be added as `.skipped(reason)` instead of ending the stream. Cancelling the stream stops after the document in flight; every stored document is complete.
- Until the embedder arrives (M2) no `.embedding` phase is emitted and chunk token counts are estimated (about four scalars per token).

### 4.3 Answers and citations

```swift
public struct Answer: Sendable {
    public let text: String                          // with inline markers like [1]; invalid markers removed, rest numbered by first use
    public let sentences: [AnsweredSentence]         // empty when `support` is nil
    public let citations: [Citation]                 // numbered in order of first use
    public let support: SupportSummary?              // overall grounding score; nil when verification did not run
    public let refused: Bool                         // true when context had no answer
    public let stats: AnswerStats
}

public struct AnsweredSentence: Sendable {
    public let text: String
    public let citationIndices: [Int]
    public let support: SupportLevel                 // .supported, .weak, .unsupported
    public let score: Float                          // 0–1 from the support scorer
}

public struct Citation: Sendable, Hashable {
    public let index: Int                            // the [n] shown to the user
    public let document: Document
    public let chunkID: ChunkID
    public let page: Int?                            // 1-based, PDFs only
    public let range: Range<Int>                     // Unicode-scalar offsets in the document's normalised text
    public let excerpt: String                       // the passage text
    public let retrievalScore: Float
}

public enum SupportLevel: String, Sendable, Codable { case supported, weak, unsupported }

public struct SupportSummary: Sendable {
    public let supportedFraction: Double             // supported sentences / all sentences
    public let level: SupportLevel                   // worst-case roll-up
}

public enum AnswerEvent: Sendable {
    case retrieved([SearchResult])                   // before generation starts — UI can show sources early
    case token(String)
    case sentence(AnsweredSentence)                  // emitted after each sentence is verified
    case finished(Answer)
}

public struct AnswerStats: Sendable {
    public let retrievalTime: Duration
    public let rerankTime: Duration
    public let timeToFirstToken: Duration
    public let generationTokensPerSecond: Double
    public let verificationTime: Duration
    public let contextTokens: Int
    public let peakMemoryBytes: UInt64
}
```

### 4.4 Search results

```swift
public struct SearchResult: Sendable, Hashable {
    public let chunkID: ChunkID
    public let document: Document
    public let page: Int?
    public let range: Range<Int>
    public let text: String
    public let scores: Scores        // .bm25, .vector, .fused, .rerank (optionals)
}
```

### 4.5 Models

```swift
public struct ModelSet: Sendable, Hashable {
    public var embedder: ModelSource
    public var reranker: ModelSource?            // nil = skip reranking
    public var generator: ModelSource?           // nil = retrieval-only knowledge base

    public static let recommended: ModelSet      // small, phone-friendly, permissively licensed
    public static let quality: ModelSet          // larger, for Macs / Pro iPhones
}

public enum ModelSource: Sendable, Hashable {
    case file(URL)
    case bundle(name: String, bundle: Bundle = .main)
    case hub(repo: String, file: String, sha256: String?)   // resolved by LociHub
}
```

`ModelSet.recommended` is bge-small-en-v1.5 Q8_0 + Qwen3-Reranker-0.6B Q8_0 + Qwen3-4B Q4_K_M; `.quality` swaps the embedder for nomic-embed-text-v1.5 Q8_0. Both pin SHA-256 checksums. The choice and the measurements behind it are in `docs/MODELS.md`.

Candidate defaults (verify GGUF availability, size, quality and **licence** before committing; record choices in `docs/MODELS.md`):

| Role | Candidates |
|------|-----------|
| Embedder | Qwen3-Embedding-0.6B, bge-small-en-v1.5, nomic-embed-text-v1.5, multilingual-e5-small |
| Reranker | Qwen3-Reranker-0.6B, bge-reranker-v2-m3 |
| Generator | Qwen3 1.7B/4B Instruct, Llama 3.2 3B Instruct, Gemma 3 4B IT |

A reranker is any GGUF with a classification head (`cls.*` tensors). Loci loads it with rank pooling even when the file's metadata omits the pooling type (some conversions do), and builds its input from the model file: models that ship a `rerank` chat template (Qwen3-Reranker) are prompted with it, others get `BOS query EOS SEP passage EOS`. The query and passage are never parsed for special tokens, and any special-token text inside them is removed before a template is tokenized, so a document cannot inject chat markers. Over-long inputs are truncated (passage first). A chat or embedding model passed as the reranker throws `modelRoleMismatch(expected: "reranker")`, checked from the file header before any weights load.

The embedding model ID and dimension are stored in the knowledge base the first time a vector is written. The ID is `<general.name>|<embedding_length>|<file size in bytes>`, read from the GGUF header, so a different model or a different quantisation of the same model counts as a different embedder.

**Opening never loads a model, and it succeeds even when the configured embedder differs from the stored one** (otherwise there would be no way to call `reindex()`). Operations that need the embedder (`add`, and `search` in `.vector` or `.hybrid` mode) throw `LociError.embedderMismatch(stored:configured:)`; `search(.keyword)`, `documents()`, `remove` and `reindex()` keep working. A knowledge base with no passages simply adopts the configured embedder. `reindex()` re-embeds every passage and records the new embedder only when it finishes; cancelling is safe and it can be re-run.

### 4.6 Options

```swift
public struct KnowledgeBaseOptions: Sendable {
    public var residency: ModelResidency          // .lowMemory (load one model at a time, default on iPhone), .balanced, .keepResident
    public var fileProtection: FileProtectionType // default .completeUntilFirstUserAuthentication
    public var gpuLayers: GPULayers               // .all default on Apple Silicon
}

public struct IngestOptions: Sendable {
    public var chunker: ChunkerKind               // .structural (default), .sentence, .tokenWindow
    public var targetChunkTokens: Int             // default 256
    public var overlapTokens: Int                 // default 32
    public var metadata: [String: String]         // stored per document, filterable
}

public struct RetrievalOptions: Sendable {
    public var limit: Int                         // final results, default 5
    public var candidatePool: Int                 // per retriever before fusion, default 40
    public var mode: RetrievalMode                // .hybrid (default), .vector, .keyword
    public var rerank: Bool                       // default true when a reranker is configured
    public var filter: MetadataFilter?            // e.g. .document(ids) or .metadata("type", "note")
}
```

`.keyword` needs no embedder; `.vector` and `.hybrid` embed the query with the configured embedder. `rerank` applies to every mode (§5.4). `SearchResult.scores.rerank` is the cross-encoder's score: for the two-way yes/no heads Loci supports it is the log-odds of "yes" (llama.cpp softmaxes that head, and the raw probabilities saturate), for single-output rerankers the raw output. Higher is better, but **scales differ between models**, so compare scores only within one reranker. Results are *passages*: overlapping or adjacent chunks of one document are merged (§5.4), so `SearchResult.range` and `text` may span several chunks and `chunkID` is the best-scoring one.

```swift
public struct KnowledgeBaseStats: Sendable, Equatable {
    public let documentCount: Int
    public let chunkCount: Int
    public let databaseBytes: UInt64              // database + WAL + SHM
}

public struct AnswerOptions: Sendable {
    public var retrieval: RetrievalOptions
    public var maxContextTokens: Int?             // nil = generator context − answer budget
    public var maxAnswerTokens: Int               // default 400
    public var temperature: Float                 // default 0.2 — grounded answers should be conservative
    public var verify: Bool                       // default true
    public var supportThresholds: (supported: Float, weak: Float)   // default (0.6, 0.35) — tuned via LociEval
    public var refuseWhenUnsupported: Bool        // default true: answer "not found in your documents" instead of guessing
    public var instructions: String?              // appended to the grounding system prompt
}
```

### 4.7 Errors

```swift
public enum LociError: Error, Sendable, LocalizedError {
    case unsupportedDocument(URL, reason: String)
    case documentUnreadable(URL, reason: String)          // encrypted PDF, scanned PDF with no text layer
    case modelNotFound(ModelSource)
    case invalidModel(ModelSource, reason: String)
    case modelRoleMismatch(ModelSource, expected: String) // "embedding" or "reranker": e.g. a chat model passed as embedder
    case embedderMismatch(stored: String, configured: String)
    case insufficientMemory(required: UInt64, available: UInt64)
    case noGenerator                                      // ask() on a retrieval-only knowledge base
    case storeCorrupted(reason: String)
    case backendFailure(operation: String, code: Int32)
    case cancelled
}
```

---

## 5. Pipeline behaviour

### 5.1 Parsing (`LociIngest`)
- **PDF:** PDFKit, page by page; keep a page → character-offset map. Pages with no text layer are counted; if all pages lack text, throw `documentUnreadable` with a hint that OCR arrives in v1.x.
- **Markdown:** keep heading hierarchy as structure; strip formatting for the stored text but record heading path per chunk.
- **HTML:** strip scripts/styles/nav; keep headings, paragraphs, list items.
- **Text:** paragraphs split on blank lines.
- All text normalised before hashing and offsetting, so offsets are stable. Each block is normalised on its own (invisible and control scalars removed, Unicode NFC, every whitespace run collapsed to one space, trimmed) and the document text is the blocks joined by a blank line (`"\n\n"`). Offsets are built while joining, never recomputed on raw text.
- **Offsets are Unicode-scalar offsets** (not `Character` or UTF-16 offsets). Chunk boundaries fall on block, sentence or whitespace boundaries, so they never split a grapheme cluster.
- Markdown is read by a built-in line parser (ATX and setext headings, lists, fenced code, tables as rows, inline formatting stripped, front matter dropped); HTML by a built-in tolerant tokenizer. Neither uses a third-party dependency.
- Text files are decoded as UTF-8, else UTF-16 with a BOM, else Latin-1.

### 5.2 Chunking
- **Structural (default):** respect headings and paragraphs; pack paragraphs up to `targetChunkTokens` using the embedder's tokenizer; split oversized paragraphs at sentence boundaries (NaturalLanguage `NLTokenizer`).
- **Sentence** and **tokenWindow** as alternatives.
- A chunk never spans a PDF page boundary, so a PDF chunk's `page` always contains its whole range. Overlap carries whole trailing units (blocks, sentences or words) of at most `overlapTokens`.
- Structural chunking starts a new chunk at every heading. Sentence packs sentences ignoring structure. Token-window packs words.
- Each chunk stores: document ID, ordinal, page (first page touched), scalar range, heading path, token count, text.

### 5.3 Store (`LociStore`)
One SQLite database per knowledge base, WAL mode, file protection applied to the DB, WAL and SHM files.

```
documents(id, title, uri, kind, page_count, content_hash UNIQUE, metadata_json, added_at,
          text, pages_json)   -- text = normalised text offsets refer to; pages_json = PDF page → range map
chunks(id, document_id, ordinal, page, start, end, heading_path, token_count, text)
chunks_fts  -- FTS5 over chunks.text (content=chunks), bm25 ranking, tokenizer: porter unicode61 remove_diacritics 2
vectors(chunk_id PRIMARY KEY → chunks ON DELETE CASCADE, embedding BLOB)   -- Float16, L2-normalised (schema v2)
meta(key, value)                    -- schema_version, embedder_id, embedder_dimension
```

- Vector search v1: brute-force cosine via Accelerate over vectors held in memory as Float16 (two bytes per component), converted to Float32 in batches of 4,096 rows and scored with one matrix-vector product per batch (`vDSP_mmul`). Float16 conversion uses vImage, because Swift's `Float16` does not exist on Intel Macs. Target: < 50 ms for 50k chunks on a recent iPhone (verify by benchmark; not yet measured).
- The in-memory copy is rebuilt lazily from SQLite after any write (insert, remove, reindex). A rebuild that overlaps another write is discarded and repeated, so a search never uses a snapshot older than the last completed write. Rows whose size does not match the query dimension (left by an interrupted reindex) are ignored.
- Each document and its vectors are written in one transaction, together with the embedder claim, so a store never holds a document without its vectors.
- v1.x: optional HNSW index when chunk count exceeds a threshold.
- Schema migrations versioned via `meta.schema_version`; a database written by a newer version is rejected.
- One writer connection and one WAL reader connection, each confined to its own actor. (SPEC calls for a small reader pool; one reader is enough until concurrent searches are measured.)
- Keyword queries are built from the question's alphanumeric terms, quoted and joined with `OR`, so user input can never be interpreted as FTS5 syntax.

### 5.4 Retrieval (`LociRetrieval`)
1. Run BM25 (FTS5) and vector search in parallel, `candidatePool` each (raised to `limit` if smaller). Each retriever honours the document filter. `.keyword` and `.vector` modes run only their retriever; `.hybrid` without a query vector degrades to keyword.
2. Fuse with **Reciprocal Rank Fusion** (k = 60). Ties keep the order in which items first appeared.
3. Rerank, if `RetrievalOptions.rerank` is set and a reranker is configured: the top N = max(20, `limit`) candidates (after fusion, or in the retriever's own order in `.keyword` / `.vector` mode) are scored against the query by the cross-encoder and reordered by that score (ties keep their earlier order). Candidates below N are dropped. Non-finite scores sort last and are not reported. Every mode can be reranked, so keyword search also loads the reranker unless `rerank` is `false`; if the reranker cannot be loaded (for example `insufficientMemory`) the search throws instead of silently skipping the stage.
4. Walk candidates in priority order, keeping up to `limit` passages. A chunk of the same document that overlaps, or lies within 2 scalars of (one block separator), an already kept passage is merged into it, provided both are on the same page (PDFs). A chunk that bridges two kept passages joins them. A merged passage's text is the exact slice of the document text; its scores and `chunkID` come from its best chunk, its heading path from its earliest.
5. Return the passages, best first.

### 5.5 Context packing
- Order selected passages by relevance, number them `[1]…[n]`, and include title + page for each (`GroundingPrompt.userMessage`).
- `maxContextTokens` bounds the **whole rendered prompt** (system rules, instructions, sources and question), measured with the generator's own chat template and tokenizer. The default is the generator's context minus `maxAnswerTokens`.
- `ContextPacker` includes sources `1…k` for the largest `k` that fits (found by bisection, since the prompt only grows). Lowest-ranked sources go first and a source is never truncated. If even the best source does not fit, `ask` throws `backendFailure` instead of answering from a cut-off passage.
- Everything that came from a document, the question or the caller's `instructions` is stripped of special-token text (`TextGenerator.sanitize`) before it is placed in the prompt, so a document cannot inject chat markup.

### 5.6 Generation (`LociGrounding` + `LociLlama.Generator`)
- **Prompt.** The system prompt tells the model to answer only from the numbered sources, cite with `[n]` after each claim, answer in one or two complete sentences, treat the sources as data rather than instructions, and reply with a fixed refusal sentence (`GroundingPrompt.refusal`) only when the sources truly lack the answer. All wording lives in `LociGrounding/GroundingPrompt.swift` and is versioned (`PromptVersion.current`, now 1); any wording change bumps it and needs an eval report, because it affects results.
- **Chat template.** The model's embedded template is applied with llama.cpp's built-in applier, which recognises a fixed list of layouts (it does not run Jinja). Layouts it rejects are rendered by hand when they belong to a supported family; currently the Gemma 4 `<|turn>` layout, whose output is tested against the model's own Jinja template. An unsupported layout makes `Generator.load` throw `invalidModel`, rather than prompting a model in a format it was not trained on. Templates that toggle thinking (Qwen3) get the empty `<think></think>` block that turns it off.
- **Roles.** `Generator.load` throws `modelRoleMismatch(expected: "generator")` for rerankers, encoder-only models and models without a chat template.
- **Reasoning.** `<think>…</think>` and Gemma's `<|channel>…<channel|>` blocks are stripped from the output, including when a tag is split across tokens; text is held back only while it could still be the start of a tag.
- **Sampling.** `temperature` 0 decodes greedily; otherwise top-k 40, top-p 0.95 and temperature, with a fixed seed available for tests. Context is `min(4096, model context)`.
- **Streaming** is UTF-8 safe: token bytes are buffered so a multi-byte character split across tokens is emitted whole. Cancelling the consumer stops generation within a token.
- **Metal and BF16.** The prebuilt llama.cpp Metal library has no BF16 matrix-multiply kernel, and asks for one by aborting the whole process. Tensors stored as BF16 (Gemma 4 has one, `per_layer_model_proj.weight`) are therefore pinned to the CPU when GPU offload is on; everything else still runs on Metal.
- **Answer flow** (`KnowledgeBase.ask` / `streamAnswer`): retrieve with `AnswerOptions.retrieval` (emitting `.retrieved` before generation); with no passages, answer with the refusal without loading the generator; otherwise pack, generate (`.token`), then resolve citations (§5.7 step 1) and detect a refusal. A reply containing the refusal sentence, ignoring case and punctuation, sets `refused` and replaces the text with the canonical sentence. `ask` throws `noGenerator` first when the `ModelSet` has none. `.token` text is the model's raw stream, with its own citation numbers; `Answer.text` is the cleaned and renumbered result.
- `AnswerStats.peakMemoryBytes` is the process's peak footprint so far (it can predate the answer); `verificationTime` is zero until verification exists.

### 5.7 Verification (the core differentiator)
After generation (and only for an answer the model did not itself refuse), `AnswerVerifier` splits the reply into sentences (line by line, so list items stay separate; a stray `[1]` after a full stop stays with its sentence; list numbers stay with their item) and, for each:
1. **Parse citations** — extract `[n]` markers (including `[1, 2]` groups). Markers naming a source that does not exist are removed. A sentence whose markers were *all* invalid is `.unsupported` and is not scored: the model cited something that is not there, and it is not given a chance to be rescued by recovery.
2. **Score support** — score (sentence, cited passage) pairs; the sentence's score is the best over its cited passages. The scorer is the **reranker** (a cross-encoder already loaded for retrieval) asked a different question from retrieval's: for instruction-aware rerankers the instruction is replaced with `GroundingPrompt.verificationInstruction` ("judge whether the Document fully supports the claim…"), because the default "retrieve relevant passages" instruction rates anything on-topic as a match. The score is the probability the reranker assigns to "supported" (the logistic of its log-odds). With no reranker the fallback is embedding cosine similarity, clamped to 0–1.
3. **Recover missing citations** — if a sentence has no marker, score it against every retrieved source; if the best reaches the `supported` threshold, attach that citation (`[n]` is inserted before the closing punctuation) and count it as `recovered` in the log. A sentence with no marker that no source supports stays uncited and `.unsupported`: it carries no citation, so it can never be `supported` or `weak`.
4. **Classify** — score ≥ `supported` → `.supported`; ≥ `weak` → `.weak`; else `.unsupported`.
5. **Renumber** — citations are numbered in order of first use across the whole answer; recovered ones take part in that order.
6. **Refusal** — if `refuseWhenUnsupported` and *more than half* of the sentences are unsupported (or there are none), the answer becomes `refused = true` with the fixed refusal sentence; `citations` then lists the retrieved passages, in rank order, for the user to inspect, and `sentences`/`support` are empty.

`.sentence` events are emitted as each sentence is finished, after generation completes (overlapping verification with generation is not implemented). `Answer.support.level` is the worst sentence's level and `supportedFraction` the share of `.supported` sentences.

**What the verifier can and cannot do.** Measured against the hand-labelled `support.jsonl` (see `docs/EVALUATION.md` and `docs/reports/`), an instruction-aware reranker catches claims that are unrelated to their passage but not contradictions that stay on topic (a changed number or name scores like a supported claim), and the embedding-similarity fallback is close to chance. Treat `.unsupported` as a flag worth showing, not proof, and `supported` as the absence of a flag, not a guarantee.

The score scale depends on the scorer, so cut-offs are calibrated per model set (`loci-cli eval --calibrate`). **Calibration is a recommendation to be checked end to end, not a default**: cut-offs that maximised F1 on the labelled pairs refused 73% of correct answers, and cut-offs held to 95% precision still refused 7%, while the spec default `(0.6, 0.35)` flags only clear-cut cases and refused none (`docs/MODELS.md`). `AnswerOptions.supportThresholds` therefore keeps the default `(0.6, 0.35)`.

### 5.8 Memory & model residency (`LociLlama/ModelResidency.swift`)
A `ModelResidencyManager` (one per knowledge base) owns the loaded models. Callers borrow a model for the duration of one operation; a borrowed model is never unloaded.

- **Estimate before loading.** `MemoryEstimate` is computed from the GGUF header without loading weights: weights (file size) + KV cache (`layers × context × kv_heads × (key + value size) × 2 bytes`, zero for encoder-only architectures such as BERT) + compute scratch (32 MiB + `context × embedding × 16` bytes). It is a heuristic that errs high; `MemoryEstimateCalibrationTests` compares it with measured process footprint and fails if it under-reports.
- **Roles come first.** Whether a file can play its role is decided from its header (embedder: a stated mean/CLS/last pooling type; reranker: `cls.*` tensors; generator: a chat template, no classifier head, not encoder-only) before memory is considered, so a wrongly assigned model reports `modelRoleMismatch` even when memory is tight.
- **Check against what is left.** Available memory is `os_proc_available_memory()` on iOS and physical RAM minus the app's current footprint on macOS. If a load does not fit, idle models are evicted least-recently-used first; if it still does not fit the call throws `insufficientMemory(required:available:)` and nothing is loaded.
- `.lowMemory`: one model resident at a time. Loading a role unloads the others, waiting if one is in use (never overlapping two). The last-used model stays loaded so repeated use, such as embedding batches during ingest, does not reload. Ingest = embedder only. Ask = embedder (query) → reranker (swaps in) → generator → reranker again for verification. Slower, fits 6 GB phones.
- `.balanced`: embedder and reranker stay resident; the generator is loaded for one answer and released afterwards.
- `.keepResident`: everything stays resident (Macs).
- **Memory pressure.** A `DispatchSource` memory-pressure source unloads idle models on warning or critical events; a model in use is unloaded as soon as it is released.
- Do not nest borrows of different roles: under `.lowMemory` the inner borrow waits for the outer one.
- Docs explain the `com.apple.developer.kernel.increased-memory-limit` entitlement.
- *Not yet verified:* behaviour on a physical 6 GB iPhone (the roadmap's M3 device check). The estimator and policies are tested on macOS only. Reranking on the CPU is slow (see `docs/EVALUATION.md`); on devices it relies on Metal.

### 5.9 Concurrency
- `KnowledgeBase` is an actor. One ingest or answer runs at a time per knowledge base; search may run concurrently with ingest (SQLite WAL readers).
- All llama.cpp work (loading, embedding, later reranking and generation) runs on one process-wide serial `DispatchQueue`, off the cooperative pool. Cancelling the calling task sets a flag the work checks between batches. Tokenizing is the one exception: it only reads the immutable vocabulary and is called from the chunker on any thread.
- Cancelling the calling task stops parsing, embedding or generation promptly and leaves the store consistent (each document is committed in one transaction).

### 5.10 Privacy & security
- `Loci` and `LociEval` link no networking APIs; CI fails if `URLSession`, `Network`, or `CFNetwork` symbols appear in their sources.
- Log messages never include document text, queries or answers; only IDs, counts and timings. `os.Logger` with `privacy: .private` for any identifiers.
- All files get the configured `FileProtectionType`.
- `destroy()` removes every file for a knowledge base.

---

## 6. LociEval (evaluation harness)

Public module + `loci-cli eval`. Full design in `docs/EVALUATION.md`.

```swift
let report = try await Evaluator(knowledgeBase: kb).run(dataset: .folder(url))   // adds documents/, then evaluates
print(report.result(for: "hybrid")?.metrics.recallAt5)
try report.write(to: outputURL, format: .markdown)      // or .json
```

`Dataset.folder(url)` is a folder with `documents/`, `questions.jsonl` and an optional `config.json` (`chunkTokens`, `overlapTokens`, `chunker`); the evaluator ingests the documents first. `Dataset.jsonl(url)` is a questions file whose documents are already in the knowledge base. (`docs/EVALUATION.md` and this section previously named different entry points; both exist.) Retrieval is evaluated per mode (keyword, vector, hybrid) over answerable questions with `limit: 10`. A relevant passage is matched by file name, optional page and an optional snippet that must occur in the retrieved text (case and whitespace ignored). Citation, faithfulness and refusal metrics arrive with M5.

Metrics: Recall@k, MRR, nDCG@k (retrieval); citation precision and recall; supported-sentence fraction; refusal accuracy on unanswerable questions; latency (retrieval, rerank, TTFT, total); peak memory; tokens/s.

---

## 7. LociHub

- Download a GGUF from Hugging Face with progress, resume, and SHA-256 verification into `Application Support/Loci/Models`.
- `ModelCatalog` lists the curated model sets with size, RAM needed, licence and expected quality from `docs/MODELS.md`.
- Background `URLSession` for large downloads on iOS.

---

## 8. Testing

| Layer | Scope | Runs |
|-------|-------|------|
| Unit | Parsers (fixture files), chunk offsets, page maps, FTS queries, RRF maths, context packing, citation parsing, support classification, UTF-8 buffering | Every PR, no models |
| Integration | Ingest → search → ask → verify with tiny GGUF fixtures (CPU, `temperature: 0`, fixed seed) | Every PR |
| Evaluation | `loci-cli eval` on the fixture dataset; fail CI if recall@5 or citation precision drop more than an agreed margin vs. `main` | Every PR (small set), nightly (full set) |
| Device | Metal path, memory residency, performance on physical iPhone and Mac | Pre-release, manual |

Invariant tests that must always hold:
- every `Citation.range` slices to exactly `Citation.excerpt` in the stored document text;
- every PDF citation's page contains its range;
- re-adding an unchanged document creates no new rows.

---

## 9. Documentation deliverables

- `README.md` — hero example, why Loci, install, models, usage, privacy, FAQ.
- DocC: Getting Started; Choosing models; How verification works; Memory on iOS; Evaluating your knowledge base; Building a document reader (step-by-step tutorial using the LociReader example).
- `docs/POSITIONING.md`, `docs/EVALUATION.md`, `docs/MODELS.md`, `docs/BENCHMARKS.md` (generated by `loci-cli bench`, never hand-written).
- Swift Package Index hosting via `.spi.yml`.

---

## 10. Versioning & release

- SemVer. Pre-1.0 minors may break.
- Store schema changes include a migration and a test that opens a database from the previous version.
- Prompt changes bump `PromptVersion` and require an eval report in the PR.
- llama.cpp bumps are patch releases unless behaviour changes; eval must not regress.

---

## 11. Open questions

1. Default chunk size and overlap — decide from eval results, not intuition.
2. Ship a small multilingual default embedder, or English-first?
3. Expose a `Generator` protocol in v1 so apps can plug in Apple's Foundation Models for the answer step, or wait until v1.1?
4. Include an encrypted-store option (e.g. SQLCipher) beyond iOS Data Protection, or leave it to v1.x?
5. Public eval dataset: build our own small one from permissively licensed documents, or adapt an existing benchmark subset (check its licence)?
