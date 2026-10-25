# Loci evaluation report

- **Dataset:** eval-small — 8 documents, 49 questions (41 answerable, 8 unanswerable)
- **Embedder:** Bge Small En v1.5|384|36685152
- **Reranker:** none
- **Generator:** none
- **Device:** Mac14,9 · Version 26.6.2 (Build 25G83)
- **Generated:** 2026-10-24T08:48:40Z
- **Ingest:** 8 documents → 97 chunks in 0.4 s (structural chunks, target 64 tokens, overlap 8)

## Verification against hand labels

Scorer: embedding similarity (Bge Small En v1.5|384|36685152). 93 labelled (claim, passage) pairs. Mean score by human label: supported 0.742, weak 0.718, unsupported 0.696. Probability a supported example outscores an unsupported one: 0.664; outscores a weak one: 0.579 (0.5 is chance).

| Cut-offs | Examples | Accuracy | Unsupported precision | Unsupported recall | Unsupported F1 | Supported F1 |
|---|---|---|---|---|---|---|
| default (supported ≥ 0.600, weak ≥ 0.350) — calibration half | 48 | 39.6% | 0.0% | 0.0% | 0.000 | 0.571 |
| default — test half | 45 | 35.6% | 0.0% | 0.0% | 0.000 | 0.533 |
| **calibrated (supported ≥ 0.579, weak ≥ 0.579) — calibration half (fitted here)** | 48 | 41.7% | 100.0% | 11.1% | 0.200 | 0.562 |
| **calibrated — test half (not used for fitting)** | 45 | 37.8% | 0.0% | 0.0% | 0.000 | 0.548 |

Calibrated `weak` flags as many unsupported claims as it can while at least 95.0% of those flagged really are unsupported; when the scorer cannot separate weak from supported claims the two cut-offs are equal.

Test-half confusion (rows: human label, columns: predicted; order supported, weak, unsupported):

- supported: 17 / 0 / 0
- weak: 11 / 0 / 0
- unsupported: 17 / 0 / 0
