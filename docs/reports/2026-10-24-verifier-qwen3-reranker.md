# Loci evaluation report

- **Dataset:** eval-small — 8 documents, 49 questions (41 answerable, 8 unanswerable)
- **Embedder:** Bge Small En v1.5|384|36685152
- **Reranker:** Qwen3 Reranker 0.6B|1024|639153184
- **Generator:** none
- **Device:** Mac14,9 · Version 26.6.2 (Build 25G83)
- **Generated:** 2026-10-24T08:48:51Z
- **Ingest:** 8 documents → 97 chunks in 0.3 s (structural chunks, target 64 tokens, overlap 8)

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
