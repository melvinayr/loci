# Loci evaluation report

- **Dataset:** eval-small — 8 documents, 49 questions (41 answerable, 8 unanswerable)
- **Embedder:** Bge Small En v1.5|384|36685152
- **Reranker:** Qwen3 Reranker 0.6B|1024|639153184
- **Generator:** gemma-4-E4B-it|2560|4590807392
- **Device:** Mac14,9 · Version 26.6.2 (Build 25G83)
- **Generated:** 2026-10-24T08:35:58Z
- **Ingest:** 8 documents → 97 chunks in 5.6 s (structural chunks, target 64 tokens, overlap 8)

## Answers

Prompt v1 · 5 passages reranked · temperature 0.0 · max 300 tokens · verification on (supported ≥ 0.600, weak ≥ 0.350)

### Citations

| Metric | Value |
|---|---|
| Citation precision | 97.7% (of 43 citations) |
| Citation recall | 100.0% (of 41 answerable questions) |
| Structural validity | 100.0% (of 83 citations; must be 100%) |

### Faithfulness

| Metric | Value |
|---|---|
| Supported sentences | 95.2% (of 42) |
| Weak sentences | 4.8% |
| Unsupported rate | 0.0% |

### Refusals and answer match

| Metric | Value |
|---|---|
| Refusal accuracy | 100.0% (of 8 unanswerable questions) |
| False refusal rate | 0.0% (of 41 answerable questions) |
| Answer contains the reference | 78.0% (of 41; a simple containment check, so a correct paraphrase counts as a miss) |

### Performance

| Stage | p50 | p95 |
|---|---|---|
| Retrieval | 17 ms | |
| Rerank | 823 ms | |
| Time to first token | 2849 ms | 3473 ms |
| Verification | 128 ms | |
| Whole answer | 7057 ms | 8409 ms |

Generation 20.3 tokens/s · peak memory 830 MB (process so far) · index 796 KB
