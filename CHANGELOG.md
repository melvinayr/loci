# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · Versioning: [SemVer](https://semver.org/)

## [Unreleased]

### Added
- Initial package skeleton: `Package.swift` with all targets from SPEC §3, Swift 6 mode.
- `LlamaBackend` smoke test and `SQLiteCapabilities` FTS5 probe.
- Integration test confirming embedding pooling works with the pinned llama.cpp.
- `Scripts/update-llama.sh`, `Scripts/fetch-test-models.sh`, MIT `LICENSE`.
- M1 ingest: PDF (with page map), Markdown, HTML and plain-text parsers; text normalisation; structural, sentence and token-window chunkers with exact Unicode-scalar offsets.
- M1 store: SQLite schema v1 with migrations, FTS5 (BM25), content-hash dedupe, file protection on database, WAL and SHM, one transaction per document.
- `KnowledgeBase` with `add`, `add(contentsOf:)` (progress stream), `remove`, `documents`, keyword `search`, `stats`, `destroy`.
- `loci-cli index`, `search` and `documents`.
- M2 embeddings: `LociLlama.Embedder` (pooling from GGUF metadata, L2 normalisation, multi-sequence batching, model-family query/document prefixes), `LlamaModel`/`LlamaContext` owners, and a serial `LlamaExecutor` with cancellation. A GGUF header reader gives a cheap embedder identity without loading weights.
- M2 vectors: store schema v2 (`vectors` table, Float16 blobs), Accelerate brute-force cosine search over an in-memory snapshot, embedder ID and dimension recorded per knowledge base.
- M2 retrieval: Reciprocal Rank Fusion, overlap and adjacency merging into passages, `.keyword` / `.vector` / `.hybrid` search with metadata filters.
- `KnowledgeBase.reindex()`, `embedderMismatch` handling, `KnowledgeBaseStats.vectorCount` and `embedderID`, `unloadModels()`.
- `LociEval`: `Evaluator`, `Dataset.folder/.jsonl`, Recall@1/5/10, MRR, nDCG@10 and latency per retrieval mode, Markdown and JSON reports, regression check.
- `loci-cli eval`, `search --mode`, `reindex`, `--models`; fixture dataset `Tests/Fixtures/eval-small` and `baseline/eval.json`; CI runs the eval gate.
- M3 reranking: `LociLlama.Reranker` (rank pooling; Qwen3-style yes/no heads with their `rerank` template and classic cross-encoders; log-odds scoring; injection-safe input), `TextReranker`, and a rerank stage in `LociRetrieval` over the top 20 candidates for every retrieval mode. `SearchResult.scores.rerank` is now populated.
- M3 residency: `ModelResidencyManager` with `.lowMemory` / `.balanced` / `.keepResident`, header-based `MemoryEstimate`, `insufficientMemory` before loading, LRU eviction of idle models, borrow tracking, and unloading on memory pressure.
- `KnowledgeBaseStats.rerankerID`, `EvalMode.hybridRerank`, `loci-cli --reranker --gpu --residency --no-rerank`, `LlamaContext` unified KV so multi-sequence batches fit the context.
- M4 generation: `LociLlama.Generator` (chat template via llama.cpp with a hand-rendered Gemma 4 fallback checked against Jinja, greedy/top-k/top-p sampling, UTF-8-safe streaming, streaming removal of `<think>` and Gemma reasoning blocks, cancellation), `TextGenerator`, and `ProcessMemory`.
- M4 grounding: versioned `GroundingPrompt` (`PromptVersion` 1), `ContextPacker` (bisection over the generator's real token counts, never truncates a source), `CitationParser` (renumbers by first use, removes invalid markers), refusal detection.
- `KnowledgeBase.ask` and `streamAnswer` with `.retrieved`, `.token` and `.finished` events; `Answer`, `Citation`, `AnswerOptions`, `AnswerStats` and friends; `loci-cli ask`.
- Prompt-injection hygiene: special-token text is stripped from documents, questions and instructions before they reach the model.
- Model roles are validated from the GGUF header before any memory check; footprint calibration tests moved to their own `LociCalibrationTests` target so concurrent tests cannot skew the measurement.
- Metal safety: BF16 tensors are pinned to the CPU when offloading, because the pinned llama.cpp aborts the process on the missing BF16 Metal kernel.
- M5 verification: `AnswerVerifier` (sentence splitting, per-sentence support scoring, citation recovery, renumbering by first use, majority refusal), `.sentence` events, `Answer.support` / `sentences`, `AnswerStats.verificationTime`, and `KnowledgeBase.supportScores`, `documentText(for:)`, `pageRanges(for:)` and `groundingPromptVersion`.
- Verification asks an instruction-aware reranker whether the passage *supports* the claim (`GroundingPrompt.verificationInstruction`) instead of whether it is relevant; `TextReranker.score` takes an optional instruction. Embedding cosine similarity is the fallback without a reranker.
- `LociEval`: citation precision and recall, structural validity, faithfulness, refusal accuracy and false refusal rate, answer match and performance (`EvalSections.answers`); agreement with the hand-labelled `support.jsonl` and threshold calibration (`.verifier`); `EvalReport` regression checks for citation precision and structural validity.
- `Tests/Fixtures/eval-small/support.jsonl`: 93 hand-labelled (claim, passage) pairs split into calibration and test halves.
- `loci-cli eval --answers --verifier --calibrate --min-precision --thresholds --no-retrieval`, `ask --no-verify --keep-unsupported --thresholds --instructions`, and generated reports in `docs/reports/`.
- Invariant tests: ranges slice to chunk text, PDF pages contain chunk ranges, re-adding is a no-op.
- M6 `LociHub`: `ModelDownloader` (Hugging Face downloads with progress, resume, streaming SHA-256 verification, disk-space check, shared in-flight downloads, cancellation that keeps the partial file) and `ModelCatalog` (the curated sets with size, memory estimate and licence). Foreground downloads only; background `URLSession` for iOS is not implemented.
- `ModelSet.recommended` and `.quality` now name real models with pinned checksums (bge-small / nomic embedder, Qwen3-Reranker-0.6B, Qwen3-4B Q4_K_M); `docs/MODELS.md` records the measurements behind the choice.
- `MemoryUsage` (current and peak process footprint) and `Benchmark` in `LociEval`; `loci-cli bench` writes `docs/BENCHMARKS.md`; `loci-cli models list|download`.
- `Examples/LociReader`: SwiftUI document reader (import, ask, streamed answer with per-sentence support badges, tap a citation to open the PDF at the page with the passage highlighted).
- DocC catalogue and tutorial for `Loci`.

### Fixed
- Residency: before refusing a load as `insufficientMemory`, return freed allocator pages to the system and re-measure, so a second question on iOS is not refused because memory from the first has not been handed back yet.

### llama.cpp
- Pinned to release `b11849` (`llama-b11849-xcframework.zip`, macOS + iOS device slices only).

### Prompt
- `PromptVersion` 1: first grounding prompt. Its wording was adjusted after an informal 20-question check on the fixture set (no answer-quality metrics exist until M5, so there is no eval report yet, and those questions are not held out).

### Changed
- `ask` and `streamAnswer` verify answers by default (`AnswerOptions.verify`). A refusal now lists the retrieved passages in `citations` instead of none.
- `Answer.support` is nil only when verification did not run or the answer was refused.
- `Evaluator(modes:)` now takes `[EvalMode]` (adds `hybrid+rerank`) instead of `[RetrievalMode]`.
- Search with a configured reranker reranks by default, in every mode; pass `rerank: false` to skip it.
- Opening a knowledge base never loads a model and does not throw `embedderMismatch`; operations that need the embedder do (SPEC §4.5).
- `add` now embeds, so it needs an embedder. `search` defaults to `.hybrid`, which returns the nearest passages even when nothing is relevant (refusal logic arrives in M5).
- Dropped visionOS and Mac Catalyst: the pinned llama.cpp binary has no slice for them. iOS Simulator is likewise unsupported.
- Minimum deployment targets raised to iOS 16.4 / macOS 13.3, the floor of the pinned llama.cpp binary (SPEC §2 updated).
