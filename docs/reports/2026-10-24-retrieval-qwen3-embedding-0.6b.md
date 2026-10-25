# Loci evaluation report

- **Dataset:** eval-small — 8 documents, 49 questions (41 answerable, 8 unanswerable)
- **Embedder:** Qwen3 Embedding 0.6b|1024|639150592
- **Reranker:** Qwen3 Reranker 0.6B|1024|639153184
- **Generator:** none
- **Device:** Mac14,9 · Version 26.6.2 (Build 25G83)
- **Generated:** 2026-10-24T09:16:15Z
- **Ingest:** 8 documents → 103 chunks in 1.9 s (structural chunks, target 64 tokens, overlap 8)

## Retrieval

Over the 41 answerable questions. Recall@k is the fraction of questions with a relevant passage in the top k.

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| keyword | 90.2% | 97.6% | 100.0% | 0.938 | 0.942 | 1 | 1 |
| vector | 92.7% | 100.0% | 100.0% | 0.953 | 0.960 | 21 | 26 |
| hybrid | 92.7% | 100.0% | 100.0% | 0.963 | 0.972 | 21 | 27 |
| hybrid+rerank | 97.6% | 100.0% | 100.0% | 0.988 | 0.991 | 596 | 626 |

## Questions without a relevant passage in the top 5

- **keyword:** tr-3 (rank 7)
- **vector:** none
- **hybrid:** none
- **hybrid+rerank:** none
