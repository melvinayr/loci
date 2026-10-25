# Loci evaluation report

- **Dataset:** eval-small — 8 documents, 49 questions (41 answerable, 8 unanswerable)
- **Embedder:** Embeddinggemma 300m|768|333590944
- **Reranker:** Qwen3 Reranker 0.6B|1024|639153184
- **Generator:** none
- **Device:** Mac14,9 · Version 26.6.2 (Build 25G83)
- **Generated:** 2026-10-24T09:17:39Z
- **Ingest:** 8 documents → 102 chunks in 1.0 s (structural chunks, target 64 tokens, overlap 8)

## Retrieval

Over the 41 answerable questions. Recall@k is the fraction of questions with a relevant passage in the top k.

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| keyword | 92.7% | 97.6% | 100.0% | 0.951 | 0.951 | 1 | 1 |
| vector | 97.6% | 100.0% | 100.0% | 0.988 | 0.989 | 12 | 13 |
| hybrid | 95.1% | 100.0% | 100.0% | 0.976 | 0.975 | 13 | 13 |
| hybrid+rerank | 97.6% | 100.0% | 100.0% | 0.988 | 0.991 | 582 | 615 |

## Questions without a relevant passage in the top 5

- **keyword:** tr-3 (rank 7)
- **vector:** none
- **hybrid:** none
- **hybrid+rerank:** none
