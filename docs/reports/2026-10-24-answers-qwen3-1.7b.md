# Loci evaluation report

- **Dataset:** eval-small — 8 documents, 49 questions (41 answerable, 8 unanswerable)
- **Embedder:** Bge Small En v1.5|384|36685152
- **Reranker:** Qwen3 Reranker 0.6B|1024|639153184
- **Generator:** Qwen3 1.7B Instruct|2048|1834426016
- **Device:** Mac14,9 · Version 26.6.2 (Build 25G83)
- **Generated:** 2026-10-24T09:20:56Z
- **Ingest:** 8 documents → 97 chunks in 0.3 s (structural chunks, target 64 tokens, overlap 8)

## Answers

Prompt v1 · 5 passages reranked · temperature 0.0 · max 300 tokens · verification on (supported ≥ 0.600, weak ≥ 0.350)

### Citations

| Metric | Value |
|---|---|
| Citation precision | 100.0% (of 38 citations) |
| Citation recall | 92.7% (of 41 answerable questions) |
| Structural validity | 100.0% (of 85 citations; must be 100%) |

### Faithfulness

| Metric | Value |
|---|---|
| Supported sentences | 95.1% (of 41) |
| Weak sentences | 2.4% |
| Unsupported rate | 2.4% |

### Refusals and answer match

| Metric | Value |
|---|---|
| Refusal accuracy | 75.0% (of 8 unanswerable questions) |
| False refusal rate | 7.3% (of 41 answerable questions) |
| Answer contains the reference | 80.5% (of 41; a simple containment check, so a correct paraphrase counts as a miss) |

### Performance

| Stage | p50 | p95 |
|---|---|---|
| Retrieval | 8 ms | |
| Rerank | 595 ms | |
| Time to first token | 497 ms | 609 ms |
| Verification | 83 ms | |
| Whole answer | 1796 ms | 2147 ms |

Generation 74.2 tokens/s · peak memory 785 MB (process so far) · index 796 KB
