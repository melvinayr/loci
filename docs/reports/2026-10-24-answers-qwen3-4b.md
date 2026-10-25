# Loci evaluation report

- **Dataset:** eval-small — 8 documents, 49 questions (41 answerable, 8 unanswerable)
- **Embedder:** Bge Small En v1.5|384|36685152
- **Reranker:** Qwen3 Reranker 0.6B|1024|639153184
- **Generator:** Qwen3 4B Instruct Awq|2560|2497280256
- **Device:** Mac14,9 · Version 26.6.2 (Build 25G83)
- **Generated:** 2026-10-24T09:23:21Z
- **Ingest:** 8 documents → 97 chunks in 0.3 s (structural chunks, target 64 tokens, overlap 8)

## Answers

Prompt v1 · 5 passages reranked · temperature 0.0 · max 300 tokens · verification on (supported ≥ 0.600, weak ≥ 0.350)

### Citations

| Metric | Value |
|---|---|
| Citation precision | 100.0% (of 37 citations) |
| Citation recall | 90.2% (of 41 answerable questions) |
| Structural validity | 100.0% (of 97 citations; must be 100%) |

### Faithfulness

| Metric | Value |
|---|---|
| Supported sentences | 91.9% (of 37) |
| Weak sentences | 8.1% |
| Unsupported rate | 0.0% |

### Refusals and answer match

| Metric | Value |
|---|---|
| Refusal accuracy | 100.0% (of 8 unanswerable questions) |
| False refusal rate | 9.8% (of 41 answerable questions) |
| Answer contains the reference | 80.5% (of 41; a simple containment check, so a correct paraphrase counts as a miss) |

### Performance

| Stage | p50 | p95 |
|---|---|---|
| Retrieval | 8 ms | |
| Rerank | 616 ms | |
| Time to first token | 1458 ms | 1762 ms |
| Verification | 77 ms | |
| Whole answer | 2812 ms | 3426 ms |

Generation 49.0 tokens/s · peak memory 924 MB (process so far) · index 796 KB
