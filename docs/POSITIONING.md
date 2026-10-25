# Positioning

*Snapshot as of 2026-10-24. Re-check before quoting numbers publicly.*

## One-sentence pitch

> **Loci gives Swift apps grounded, cited answers from the user's own documents — entirely on device, with every claim checked against its source.**

## The landscape

On-device RAG in Swift exists in pieces. Each library covers one segment of the pipeline:

| | Ingest + chunk | Embeddings | Store | Hybrid search | Rerank | Generates answers | Verified citations | Native GGUF in-process | Evaluation | Licence | Min iOS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LumoKit** | ✅ | via VecturaKit | via VecturaKit | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | MIT | 18 |
| **VecturaKit** | ❌ | NL, MLX, swift-embeddings, OpenAI-compatible | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ (server only) | ❌ | MIT | 18 |
| **Wax** | Markdown sync | needs local embedder | ✅ single file, Metal HNSW | ✅ | ❌ | ❌ | ❌ | not stated | ❌ | Apache-2.0 | not stated |
| **NobodyWho** | ❌ | ✅ GGUF | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | EUPL-1.2 (copyleft) | not stated |
| **LLM.swift** | ❌ | basic | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | MIT | — |
| **Loci** | ✅ | ✅ GGUF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | MIT | **16** |

Today a developer wanting "ask questions about my documents, offline" must combine three libraries and hand-write prompt assembly, context budgeting and citation tracking.

## What is unique — in order of defensibility

1. **Verified citations.** Every answer sentence is linked to source passages with page and character offsets, then scored for support by a cross-encoder. Unsupported claims are flagged or the answer is refused. This is hard to bolt onto a storage library later because it requires offsets, original text and a reranker designed in from the start.
2. **Measured quality.** `LociEval` reports retrieval recall, citation precision/recall, faithfulness, latency and memory per device. Competitors show that retrieval works; Loci shows *how well*.
3. **The complete loop.** Documents in, cited answer out, one package, one API call.
4. **One native GGUF stack under one memory budget.** Embedder, reranker and generator run in-process via llama.cpp + Metal, with residency policies that make all three fit on a 6 GB iPhone.
5. **Reach and licence.** iOS 16+ (vs. 18+), MIT licensed (vs. copyleft for the closest full-stack alternative).

Items 4 and 5 are real but easy to copy; lead with 1 and 2.

## Who it's for

Apps where a wrong or untraceable answer is unacceptable and data can't leave the device: health and clinical notes, research literature, legal and compliance documents, enterprise field manuals, personal journals.

## Interview answer (30 seconds)

> "On-device RAG in Swift exists in pieces — VecturaKit and Wax handle storage and search, LumoKit handles ingestion, LLM.swift handles generation — but nobody connects them into an answer you can trust. Loci does the full loop natively on llama.cpp: documents go in, and an answer comes out where every sentence is cited to a source passage and checked by a cross-encoder for support. It ships with an evaluation harness, so accuracy is measured rather than assumed. It's built for apps with sensitive documents, where nothing can leave the device and a made-up source is unacceptable."

## Likely follow-up questions

**"Why not build on VecturaKit?"** Citations need exact page and character offsets stored next to the original text, verification needs the text co-located with vectors, and VecturaKit requires iOS 18. Owning a single SQLite file also makes Data Protection and deletion simple. A VecturaKit adapter can come later.

**"How do you know the citations are right?"** Two layers: structural invariants tested in CI (every citation range slices to its excerpt, every page contains its range), and semantic support scoring with a cross-encoder, calibrated and reported by `LociEval`.

**"What about Apple's Foundation Models?"** It's a strong generator on supported devices. Loci's value is the retrieval, grounding and verification around generation; a `Generator` protocol lets Apple's model be plugged in for the answer step.

**"Won't llama.cpp API changes break you?"** It's pinned to a release XCFramework and only one internal target imports it. A weekly bot PR bumps it, and the eval suite must not regress before merging.

**"What's the hardest part technically?"** Fitting three models into a phone's memory budget while keeping latency acceptable, and calibrating support thresholds so verification flags real hallucinations without rejecting good answers.

## Sources
- LumoKit — https://github.com/rryam/LumoKit
- VecturaKit — https://github.com/rryam/VecturaKit
- Wax — https://github.com/christopherkarani/Wax
- NobodyWho — https://docs.nobodywho.ooo/swift/embeddings-and-rag
- LLM.swift — https://github.com/eastriverlee/LLM.swift
