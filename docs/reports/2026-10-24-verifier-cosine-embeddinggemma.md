# Loci evaluation report

- **Dataset:** eval-small — 8 documents, 49 questions (41 answerable, 8 unanswerable)
- **Embedder:** Embeddinggemma 300m|768|333590944
- **Reranker:** none
- **Generator:** none
- **Device:** Mac14,9 · Version 26.6.2 (Build 25G83)
- **Generated:** 2026-10-24T08:48:44Z
- **Ingest:** 8 documents → 102 chunks in 1.0 s (structural chunks, target 64 tokens, overlap 8)

## Verification against hand labels

Scorer: embedding similarity (Embeddinggemma 300m|768|333590944). 93 labelled (claim, passage) pairs. Mean score by human label: supported 0.586, weak 0.535, unsupported 0.518. Probability a supported example outscores an unsupported one: 0.700; outscores a weak one: 0.636 (0.5 is chance).

| Cut-offs | Examples | Accuracy | Unsupported precision | Unsupported recall | Unsupported F1 | Supported F1 |
|---|---|---|---|---|---|---|
| default (supported ≥ 0.600, weak ≥ 0.350) — calibration half | 48 | 37.5% | 100.0% | 11.1% | 0.200 | 0.452 |
| default — test half | 45 | 35.6% | 0.0% | 0.0% | 0.000 | 0.533 |
| **calibrated (supported ≥ 0.387, weak ≥ 0.387) — calibration half (fitted here)** | 48 | 43.8% | 100.0% | 16.7% | 0.286 | 0.571 |
| **calibrated — test half (not used for fitting)** | 45 | 40.0% | 40.0% | 11.8% | 0.182 | 0.561 |

Calibrated `weak` flags as many unsupported claims as it can while at least 95.0% of those flagged really are unsupported; when the scorer cannot separate weak from supported claims the two cut-offs are equal.

Test-half confusion (rows: human label, columns: predicted; order supported, weak, unsupported):

- supported: 16 / 0 / 1
- weak: 9 / 0 / 2
- unsupported: 15 / 0 / 2
