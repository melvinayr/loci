# Loci evaluation report

- **Dataset:** eval-small — 8 documents, 49 questions (41 answerable, 8 unanswerable)
- **Embedder:** Bge Small En v1.5|384|36685152
- **Reranker:** Qwen3 Reranker 0.6B|1024|639153184
- **Generator:** gemma-4-E4B-it|2560|4590807392
- **Device:** Mac14,9 · Version 26.6.2 (Build 25G83)
- **Generated:** 2026-10-24T08:47:28Z
- **Ingest:** 8 documents → 97 chunks in 0.3 s (structural chunks, target 64 tokens, overlap 8)

## Answers

Prompt v1 · 5 passages reranked · temperature 0.0 · max 300 tokens · verification on (supported ≥ 0.999, weak ≥ 0.844)

### Citations

| Metric | Value |
|---|---|
| Citation precision | 97.5% (of 40 citations) |
| Citation recall | 92.7% (of 41 answerable questions) |
| Structural validity | 100.0% (of 95 citations; must be 100%) |

### Faithfulness

| Metric | Value |
|---|---|
| Supported sentences | 10.3% (of 39) |
| Weak sentences | 89.7% |
| Unsupported rate | 0.0% |

### Refusals and answer match

| Metric | Value |
|---|---|
| Refusal accuracy | 100.0% (of 8 unanswerable questions) |
| False refusal rate | 7.3% (of 41 answerable questions) |
| Answer contains the reference | 75.6% (of 41; a simple containment check, so a correct paraphrase counts as a miss) |

### Performance

| Stage | p50 | p95 |
|---|---|---|
| Retrieval | 9 ms | |
| Rerank | 604 ms | |
| Time to first token | 1421 ms | 1734 ms |
| Verification | 81 ms | |
| Whole answer | 3626 ms | 4244 ms |

Generation 35.3 tokens/s · peak memory 826 MB (process so far) · index 796 KB

## Verification against hand labels

Scorer: reranker Qwen3 Reranker 0.6B|1024|639153184. 93 labelled (claim, passage) pairs. Mean score by human label: supported 0.998, weak 0.988, unsupported 0.742. Probability a supported example outscores an unsupported one: 0.869; outscores a weak one: 0.791 (0.5 is chance).

| Cut-offs | Examples | Accuracy | Unsupported precision | Unsupported recall | Unsupported F1 | Supported F1 |
|---|---|---|---|---|---|---|
| default (supported ≥ 0.600, weak ≥ 0.350) — calibration half | 48 | 47.9% | 100.0% | 27.8% | 0.435 | 0.590 |
| default — test half | 45 | 44.4% | 100.0% | 17.6% | 0.300 | 0.586 |
| **calibrated (supported ≥ 0.999, weak ≥ 0.844) — calibration half (fitted here)** | 48 | 50.0% | 100.0% | 33.3% | 0.500 | 0.500 |
| **calibrated — test half (not used for fitting)** | 45 | 48.9% | 100.0% | 41.2% | 0.583 | 0.364 |

Calibrated `weak` flags as many unsupported claims as it can while at least 95.0% of those flagged really are unsupported; when the scorer cannot separate weak from supported claims the two cut-offs are equal.

Test-half confusion (rows: human label, columns: predicted; order supported, weak, unsupported):

- supported: 4 / 13 / 0
- weak: 0 / 11 / 0
- unsupported: 1 / 9 / 7
