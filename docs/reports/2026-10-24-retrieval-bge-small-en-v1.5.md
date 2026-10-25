# Loci evaluation report

- **Dataset:** eval-small — 8 documents, 49 questions (41 answerable, 8 unanswerable)
- **Embedder:** Bge Small En v1.5|384|36685152
- **Reranker:** Qwen3 Reranker 0.6B|1024|639153184
- **Generator:** none
- **Device:** Mac14,9 · Version 26.6.2 (Build 25G83)
- **Generated:** 2026-10-24T09:17:11Z
- **Ingest:** 8 documents → 97 chunks in 0.3 s (structural chunks, target 64 tokens, overlap 8)

## Retrieval

Over the 41 answerable questions. Recall@k is the fraction of questions with a relevant passage in the top k.

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| keyword | 92.7% | 97.6% | 100.0% | 0.949 | 0.949 | 1 | 1 |
| vector | 92.7% | 100.0% | 100.0% | 0.953 | 0.960 | 7 | 8 |
| hybrid | 90.2% | 100.0% | 100.0% | 0.947 | 0.959 | 8 | 8 |
| hybrid+rerank | 97.6% | 100.0% | 100.0% | 0.988 | 0.991 | 602 | 641 |

## Questions without a relevant passage in the top 5

- **keyword:** tr-3 (rank 7)
- **vector:** none
- **hybrid:** none
- **hybrid+rerank:** none
