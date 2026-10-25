# Architecture

## Layers

```
┌──────────────────────────────────────────────────────────────────┐
│ App                                                              │
├──────────────────────────────────────────────────────────────────┤
│ Loci (public)      KnowledgeBase · Answer · Citation · Options   │
├───────────────┬───────────────┬────────────────┬─────────────────┤
│ LociIngest    │ LociRetrieval │ LociGrounding  │ LociEval (pub)  │
│ parse, chunk  │ hybrid, RRF,  │ prompt, cite   │ metrics,        │
│ offsets/pages │ rerank, pack  │ parse, verify  │ calibration     │
├───────────────┴───────┬───────┴────────────────┴─────────────────┤
│ LociStore             │ LociLlama (only llama.cpp importer)      │
│ SQLite · FTS5 ·       │ Embedder · Reranker · Generator ·        │
│ Float16 vectors       │ ModelResidency                           │
├───────────────────────┼──────────────────────────────────────────┤
│ SQLite                │ llama.xcframework (Metal, Accelerate)    │
└───────────────────────┴──────────────────────────────────────────┘
  LociHub (pub, optional) — the only target with network access
```

Dependency rule: arrows point down only. `LociGrounding` may use `LociLlama` (for scoring) and `LociRetrieval` types, never `LociStore` directly.

## Ingest flow

```
add(file)
 → parse (PDFKit / Markdown / HTML / text) → normalised text + page map
 → contentHash; unchanged? → .skipped (nothing is chunked or embedded)
 → embedder identity matches the store?  else embedderMismatch
 → chunk (structural, sized with the embedder's tokenizer) → [Chunk(range, page, headingPath, text)]
 → embed in batches (Embedder)        ← model residency may load the embedder here
 → one SQLite transaction: document + chunks + FTS rows + vectors + embedder claim
```

## Answer flow

```
ask(question)
 1. embed query                        (Embedder)
 2. BM25 ∥ vector search               (LociStore)
 3. RRF fuse → top N                   (LociRetrieval)
 4. rerank → top k                     (Reranker)          → .retrieved event
 5. pack numbered context              (LociGrounding)     fits the generator's context, best sources first
 6. generate with [n] citations        (Generator)         → .token events
 7. detect a refusal sentence; otherwise verify:               (AnswerVerifier)
    split into sentences → resolve [n] markers → score each sentence against its cited
    passages with the Reranker, asked to judge support → recover missing citations →
    classify → renumber by first use                         → .sentence events
 8. majority unsupported? refuse; else assemble Answer        → .finished
```

In `.lowMemory` residency, the Embedder is unloaded after step 1, the Reranker stays for steps 4 and 7, and the Generator is loaded for step 6. Verification of sentence n can overlap with generation of sentence n+1 only in `.keepResident`.

## Model residency

Every model use goes through one `ModelResidencyManager` per knowledge base: models load lazily, are checked against available memory first, and are kept or released by the residency policy (see SPEC §5.8). Ingest borrows the embedder for chunking (its tokenizer sizes the chunks) and embedding; search borrows it for the query, then the reranker for the top candidates. In `.lowMemory` those two borrows swap the models rather than overlapping them.

## Why the reranker verifies

A cross-encoder scores how well a passage matches a query. Feeding it (answer sentence, cited passage) gives a cheap support signal from a model that is already loaded for retrieval: no separate NLI model, no extra memory. Asked its default question ("is this passage relevant?") it rates anything on-topic highly, so verification replaces the instruction of an instruction-aware reranker with one that asks whether the passage *supports* the claim; that raised the separation of supported from unsupported claims noticeably. It remains a relevance model at heart: it catches claims unrelated to their passage and misses on-topic contradictions. Score scales differ between rerankers, so `LociEval` measures agreement with hand labels and fits cut-offs per model set.

## Ownership of C resources

| Swift type | Owns | Freed |
|---|---|---|
| `LlamaBackend` | backend init | process lifetime |
| `LlamaModel` | `llama_model *` | `deinit` |
| `LlamaContext` | `llama_context *`, batch | `deinit` (holds its `LlamaModel` strongly) |
| `SQLiteConnection` | `sqlite3 *`, prepared statements | `deinit` |

## Threading

- `KnowledgeBase` actor serialises writes and answers.
- llama.cpp work runs on one process-wide serial queue (`LlamaExecutor`), off the cooperative pool.
- SQLite: one writer connection, a small pool of read connections (WAL), each confined to its executor.
- All streams use `AsyncThrowingStream` with `onTermination` cancelling the producer.

## Key decisions

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Inference engine | llama.cpp release XCFramework | MLX, Core ML, source build | GGUF ecosystem breadth (embedders, rerankers, chat), iOS 16 reach, no C++ interop for users |
| Storage | Own SQLite + FTS5 + Float16 blobs | VecturaKit, Wax | Citation offsets first-class, iOS 16, single protected file |
| Vector search | Brute force (Accelerate) | HNSW in v1 | Exact results, simple, fast enough for personal-scale collections; HNSW later |
| Fusion | Reciprocal Rank Fusion | Weighted score blend | No score normalisation needed across BM25 and cosine |
| Verification | Reranker cross-encoder | Separate NLI model, LLM-as-judge | No extra memory, deterministic, fast |
