# Loci evaluation report

- **Dataset:** eval-small — 8 documents, 49 questions (41 answerable, 8 unanswerable)
- **Embedder:** Bge Small En v1.5|384|36685152
- **Reranker:** Qwen3 Reranker 0.6B|1024|639153184
- **Generator:** Qwen3 0.6B Instruct|1024|639446688
- **Device:** Mac14,9 · Version 26.6.2 (Build 25G83)
- **Generated:** 2026-10-24T09:19:24Z
- **Ingest:** 8 documents → 97 chunks in 3.6 s (structural chunks, target 64 tokens, overlap 8)

## Answers

Prompt v1 · 5 passages reranked · temperature 0.0 · max 300 tokens · verification on (supported ≥ 0.600, weak ≥ 0.350)

### Citations

| Metric | Value |
|---|---|
| Citation precision | 100.0% (of 36 citations) |
| Citation recall | 87.8% (of 41 answerable questions) |
| Structural validity | 100.0% (of 101 citations; must be 100%) |

### Faithfulness

| Metric | Value |
|---|---|
| Supported sentences | 97.3% (of 37) |
| Weak sentences | 0.0% |
| Unsupported rate | 2.7% |

### Refusals and answer match

| Metric | Value |
|---|---|
| Refusal accuracy | 100.0% (of 8 unanswerable questions) |
| False refusal rate | 12.2% (of 41 answerable questions) |
| Answer contains the reference | 78.0% (of 41; a simple containment check, so a correct paraphrase counts as a miss) |

### Performance

| Stage | p50 | p95 |
|---|---|---|
| Retrieval | 8 ms | |
| Rerank | 592 ms | |
| Time to first token | 199 ms | 246 ms |
| Verification | 76 ms | |
| Whole answer | 1277 ms | 1451 ms |

Generation 143.7 tokens/s · peak memory 772 MB (process so far) · index 796 KB
