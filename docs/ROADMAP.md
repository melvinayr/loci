# Roadmap

Each milestone is sized for one focused Claude Code session and ends with something demonstrable. Start each session with:
> "Read CLAUDE.md and docs/SPEC.md. Implement Milestone N from docs/ROADMAP.md. Stop and ask if anything in the spec is ambiguous."

## M0 — Foundations
- `Package.swift`: targets and products from SPEC §3, Swift 6 mode, platforms from SPEC §2
- llama.cpp `binaryTarget` (pinned release + checksum); smoke test of backend init in `LociLlama`
- **Verify** FTS5 in system SQLite on iOS 16 / macOS 13; vendor the amalgamation if missing (SPEC §3.2)
- **Verify** the pinned llama.cpp release supports embedding pooling and rank pooling
- CI: build, unit tests, `import llama` only in `LociLlama`, no networking symbols in `Loci`/`LociEval`
- `.gitignore`, MIT `LICENSE`, `.spi.yml`
- **Demo:** green CI

## M1 — Ingest & store
- Parsers: PDF (with page map), Markdown, HTML, plain text; normalisation
- Structural, sentence and token-window chunkers with offsets
- `LociStore`: schema, migrations, FTS5, file protection, transactions, content-hash dedupe
- `KnowledgeBase.add`, `add(contentsOf:)` with progress, `remove`, `documents`
- Invariant tests: offsets slice to text; pages contain ranges; re-add is a no-op
- **Demo:** `loci-cli index ./docs` then keyword search returns passages with page numbers

## M2 — Embeddings & hybrid retrieval
- `LociLlama.Embedder` (pooling from GGUF metadata, normalisation, batching)
- Vector storage (Float16) and Accelerate brute-force search
- RRF fusion, dedupe/merge, metadata filters, `search()`
- Embedder ID/dimension stored; mismatch error; `reindex()`
- `LociEval` retrieval metrics + fixture dataset v0 (documents + questions)
- **Demo:** eval report comparing keyword vs vector vs hybrid Recall@5

## M3 — Reranking
- `LociLlama.Reranker` (rank pooling), rerank stage in retrieval
- `ModelResidency` with memory estimation, `.lowMemory`/`.balanced`/`.keepResident`, memory-pressure handling
- **Demo:** eval shows hybrid + rerank improvement; runs on a physical 6 GB iPhone

## M4 — Grounded generation
- `LociLlama.Generator` (chat template, sampler, UTF-8-safe streaming, thinking-block stripping)
- Context packing with numbered sources; versioned grounding prompt
- `ask()` / `streamAnswer()` with `.retrieved`, `.token`, `.finished` events
- **Demo:** `loci-cli ask "…"` prints an answer with [n] markers and a source list

## M5 — Verification (the differentiator)
- Citation parsing, invalid-marker removal, citation recovery
- Support scoring via reranker (cosine fallback), sentence classification, refusal logic
- `.sentence` streaming events
- Citation precision/recall, faithfulness and refusal metrics in `LociEval`; threshold calibration
- Hand-labelled support subset in the fixture dataset
- **Demo:** an answer where a deliberately unsupported sentence is flagged; eval report with faithfulness numbers

## M6 — Example app & docs (v0.1.0 public)
- `LociReader` SwiftUI app: import files, ask, stream answer, tap a citation → open PDF at the page with the passage highlighted, support badges per sentence
- `LociHub` downloads + `ModelCatalog`; `docs/MODELS.md` with chosen defaults, licences, calibrated thresholds
- DocC catalogue + tutorial; Swift Package Index hosting
- `loci-cli bench` → `docs/BENCHMARKS.md`
- README GIF of LociReader (a citation tap is the money shot)
- **Launch:** Swift Forums, r/LocalLLaMA, r/iOSProgramming, iOS Dev Weekly, LinkedIn; short write-up of the evaluation results

## v0.2+
- OCR for scanned PDFs (Vision)
- HNSW index for large collections
- `Generator` protocol with an Apple Foundation Models adapter
- Encrypted store option
- Incremental folder watching
- Conversation follow-ups with query rewriting
- SwiftUI components (`CitationView`, `AnswerView`)
- Optional VecturaKit adapter
- Technical paper / blog on on-device citation verification results
