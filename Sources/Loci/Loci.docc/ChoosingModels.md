# Choosing models

Which models Loci uses, and why.

## The three roles

A knowledge base needs an embedder, and optionally a reranker and a generator. Without a generator it is a retrieval-only search engine; without a reranker, verification falls back to embedding similarity, which is close to chance, so configure one if you show support badges.

## The shipped sets

``ModelSet/recommended`` uses bge-small-en-v1.5, Qwen3-Reranker-0.6B and Qwen3-4B (Q4_K_M). ``ModelSet/quality`` swaps in the larger nomic-embed-text-v1.5 embedder. Sizes, licences and the measurements behind the choice are in `docs/MODELS.md` in the repository, with the reports under `docs/reports`.

Two findings worth knowing:

- A smaller generator is not automatically better value. Qwen3-1.7B was faster but answered two of eight unanswerable questions instead of declining; Qwen3-4B declined all eight.
- After reranking, the embedders tested were indistinguishable on the evaluation set, so the smallest, bge-small, is the default.

## English first

The shipped embedders are English-only. Other languages need a multilingual embedder; Loci accepts any GGUF embedder, but its quality on non-English documents has not been measured.

## Bringing your own

Use `.file(url)` sources. Loci reads the GGUF header to confirm each file fits its role before loading it, and records the embedder in the store: switching embedders requires ``KnowledgeBase/reindex()``.
