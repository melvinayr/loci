# Loci evaluation report

- **Dataset:** eval-small — 8 documents, 49 questions (41 answerable, 8 unanswerable)
- **Embedder:** Bge Small En v1.5|384|36685152
- **Reranker:** Qwen3 Reranker 0.6B|1024|639153184
- **Generator:** gemma-4-E4B-it|2560|4590807392
- **Device:** Mac14,9 · Version 26.6.2 (Build 25G83)
- **Generated:** 2026-10-24T08:41:00Z
- **Ingest:** 8 documents → 97 chunks in 1.3 s (structural chunks, target 64 tokens, overlap 8)

## Answers

Prompt v1 · 5 passages reranked · temperature 0.0 · max 300 tokens · verification on (supported ≥ 0.997, weak ≥ 0.996)

### Citations

| Metric | Value |
|---|---|
| Citation precision | 92.3% (of 13 citations) |
| Citation recall | 26.8% (of 41 answerable questions) |
| Structural validity | 100.0% (of 203 citations; must be 100%) |

### Faithfulness

| Metric | Value |
|---|---|
| Supported sentences | 91.7% (of 12) |
| Weak sentences | 8.3% |
| Unsupported rate | 0.0% |

### Refusals and answer match

| Metric | Value |
|---|---|
| Refusal accuracy | 100.0% (of 8 unanswerable questions) |
| False refusal rate | 73.2% (of 41 answerable questions) |
| Answer contains the reference | 22.0% (of 41; a simple containment check, so a correct paraphrase counts as a miss) |

### Performance

| Stage | p50 | p95 |
|---|---|---|
| Retrieval | 13 ms | |
| Rerank | 723 ms | |
| Time to first token | 2452 ms | 3407 ms |
| Verification | 111 ms | |
| Whole answer | 6216 ms | 8082 ms |

Generation 24.2 tokens/s · peak memory 831 MB (process so far) · index 796 KB

## Verification against hand labels

Scorer: reranker Qwen3 Reranker 0.6B|1024|639153184. 93 labelled (claim, passage) pairs. Mean score by human label: supported 0.998, weak 0.988, unsupported 0.742. Probability a supported example outscores an unsupported one: 0.869 (0.5 is chance).

| Cut-offs | Examples | Accuracy | Unsupported precision | Unsupported recall | Unsupported F1 | Supported F1 |
|---|---|---|---|---|---|---|
| default (supported ≥ 0.600, weak ≥ 0.350) — calibration half | 48 | 47.9% | 100.0% | 27.8% | 0.435 | 0.590 |
| default — test half | 45 | 44.4% | 100.0% | 17.6% | 0.300 | 0.586 |
| **calibrated (supported ≥ 0.997, weak ≥ 0.996) — calibration half (fitted here)** | 48 | 66.7% | 66.7% | 77.8% | 0.718 | 0.773 |
| **calibrated — test half (not used for fitting)** | 45 | 57.8% | 66.7% | 70.6% | 0.686 | 0.651 |

Test-half confusion (rows: human label, columns: predicted; order supported, weak, unsupported):

- supported: 14 / 0 / 3
- weak: 8 / 0 / 3
- unsupported: 4 / 1 / 12
