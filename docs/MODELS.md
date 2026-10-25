# Models

Every size and licence below was checked against the model's Hugging Face repository (file sizes and SHA-256 from the repository's file listing, licences from the repository metadata). Every measured number comes from a generated report in `docs/reports/`; none was typed in by hand. The evaluation dataset (`eval-small`: 8 documents, 49 questions) is small, so treat differences of a few points as noise.

## Model sets

Both sets are in `ModelSet` (and described by `ModelCatalog` in `LociHub`) and pin SHA-256 checksums.

### `ModelSet.recommended` (iPhone 6 GB+, Mac)

| Role | Model | Quant | Size | Licence | Source |
|---|---|---|---|---|---|
| Embedder | bge-small-en-v1.5 | Q8_0 | 37 MB | MIT | `ggml-org/bge-small-en-v1.5-Q8_0-GGUF` |
| Reranker | Qwen3-Reranker-0.6B | Q8_0 | 639 MB | Apache-2.0 | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` |
| Generator | Qwen3-4B | Q4_K_M | 2.5 GB | Apache-2.0 | `Qwen/Qwen3-4B-GGUF` |

About 3.2 GB on disk. In `.lowMemory` residency only one model is loaded at a time, so the peak is the generator's size plus buffers (the catalogue estimates a fifth on top of the file; that is an estimate, not a measurement).

### `ModelSet.quality` (Mac, Pro iPhones)

The same reranker and generator, with nomic-embed-text-v1.5 Q8_0 (146 MB, Apache-2.0, `nomic-ai/nomic-embed-text-v1.5-GGUF`) as embedder.

## Why these models

**Generator.** Answer reports with the shipped thresholds, Metal, same retrieval, reranker and prompt (prompt v1):

| Generator | Refusal accuracy (8 unanswerable) | False refusals (41 answerable) | Citation precision / recall | Answer contains reference | Generation speed |
|---|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | 100% | 12.2% | 100% / 87.8% | 78% | 144 tok/s |
| Qwen3-1.7B Q8_0 | **75%** (answered 2 of 8) | 7.3% | 100% / 92.7% | 80.5% | 74 tok/s |
| **Qwen3-4B Q4_K_M** | 100% | 9.8% | 100% / 90.2% | 80.5% | 49 tok/s |
| Gemma 4 E4B Q4_0 | 100% | 0% | 97.7% / 100% | 78% | 20 tok/s |

Reports: `2026-10-24-answers-qwen3-0.6b.md`, `…-1.7b.md`, `…-4b.md`, `2026-10-24-gemma4-qwen3reranker-default-thresholds.md`. Qwen3-1.7B is faster but invented answers to two questions the documents cannot answer, which is the failure Loci exists to prevent, so it is not the default. Qwen3-4B is the smallest model that refused every unanswerable question. Gemma 4 E4B scored best on false refusals and citation recall, but its licence has not been verified and it is the slowest, so it is not recommended; it remains usable by pointing a `ModelSet` at the file. "Answer contains reference" is a simple containment check, so correct paraphrases count as misses.

**Embedder.** All four candidates tie once the reranker is applied (Recall@1 97.6%, MRR 0.988): the dataset is saturated and cannot separate them after reranking. Before reranking embeddinggemma-300m has the best vector results, but it is under the Gemma terms, larger, and not better after reranking. bge-small is the smallest and MIT-licensed, hence the default; nomic-embed-text-v1.5 is the larger Apache-2.0 option. Reports: `2026-10-24-retrieval-*.md`.

**Reranker.** Qwen3-Reranker-0.6B is the only one tested for verification (below) and doubles as the support checker, so it is in both sets.

**English first.** Both embedders are English-only. Multilingual embedders (Qwen3-Embedding-0.6B is Apache-2.0 and ran in the retrieval comparison, on English questions only) have not been evaluated on non-English documents; that remains open.

## Verified reranker files (used by tests)

Facts about files Loci has been run against.

| File | Source | Licence | Size | Notes |
|---|---|---|---|---|
| `qwen3-reranker-0.6b-q8_0.gguf` | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` | Apache-2.0 | 639 MB | Ships rank pooling, yes/no classifier labels and a `rerank` chat template. Loads and scores correctly. |
| `bge-reranker-v2-m3-Q4_K_M.gguf` | `gpustack/bge-reranker-v2-m3-GGUF` | Apache-2.0 | 438 MB | Omits the pooling type; Loci forces rank pooling because the file has a classification head. Scores with the `BOS query EOS SEP passage EOS` layout. |

## Verified generator files (used by tests)

| File | Source | Licence | Size | Notes |
|---|---|---|---|---|
| `qwen3-0.6b-q8_0.gguf` | `Qwen/Qwen3-0.6B-GGUF` | Apache-2.0 | 639 MB | ChatML template recognised by llama.cpp; thinks by default, which Loci switches off. Small enough for CI; follows the grounding prompt only roughly. |
| `gemma-4-E4B-it-Q4_0.gguf` | supplied locally for testing | not verified | 4.6 GB | Its chat template is not recognised by llama.cpp's built-in applier (Loci renders it by hand). Contains one BF16 tensor, which Loci keeps on the CPU because the pinned Metal library has no BF16 kernel. Not distributed with Loci. |

## Not yet evaluated

multilingual-e5-small, bge-reranker-v2-m3 for verification, Llama 3.2 3B Instruct, Gemma 3 4B IT.

## Calibrated support thresholds

Verification scores each answer sentence 0–1 (see the “How verification works” article in the DocC catalogue). Numbers below come from the generated reports in `docs/reports/` (dataset `eval-small`, 93 hand-labelled pairs split 48 calibration / 45 test, 2026-10-24); the generator for the end-to-end rows is Gemma 4 E4B on Metal. Treat differences of a few points as noise: the dataset is small.

**Qwen3-Reranker-0.6B (Q8_0), instruction-aware verification.** Probability that a supported claim outscores an unsupported one: 0.87 (0.5 is chance); a weak one: 0.79. Unsupported claims split in two: unrelated ones score below 0.42 and are caught, but about two thirds score above 0.94, indistinguishable from supported claims.

| Cut-offs (supported, weak) | Unsupported flag, test half: precision / recall | End to end, 41 answerable questions: false refusals | Sentences shown as `supported` | Report |
|---|---|---|---|---|
| **(0.6, 0.35) — shipped default** | 100% / 18% | 0% | 95% | `2026-10-24-gemma4-qwen3reranker-default-thresholds.md` |
| (0.997, 0.996) — fitted to maximise F1 | 67% / 71% | **73%** — rejected | 92% (of the few answers left) | `…-f1-calibrated-REJECTED.md` |
| (0.999, 0.844) — fitted with a 95% precision floor | 100% / 41% | 7% (three correct one-sentence answers scored 0.41–0.71) | 10% (the rest shown as `weak`) | `…-precision-calibrated.md` |

The default flags only clear-cut cases; the fitted cut-offs flag more unsupported claims but pay for it with false refusals of correct answers. Reproduce with `loci-cli eval --models fixture --reranker fixture --generator <gguf> --gpu --no-retrieval --answers --calibrate`.

**Embedding-similarity fallback (no reranker).** Close to chance, so its labels should not drive decisions: AUC 0.66 with bge-small and 0.70 with embeddinggemma; any fitted cut-off flags most claims or few of the unsupported ones (`2026-10-24-verifier-cosine-bge-small.md`, `…-embeddinggemma.md`). Configure a reranker for meaningful verification. The default `(0.6, 0.35)` flags nothing on these scores, which is the safe failure.

| Model set | Reranker | supported ≥ | weak ≥ | Notes |
|---|---|---|---|---|
| any with Qwen3-Reranker-0.6B | Qwen3-Reranker-0.6B Q8_0 | 0.6 (default) | 0.35 (default) | Conservative; see above. |
| `recommended`, `quality` | Qwen3-Reranker-0.6B Q8_0 | 0.6 (default) | 0.35 (default) | Same as above. |
| a set without a reranker | none | 0.6 (default) | 0.35 (default) | Cosine fallback: effectively unverified. |

## Reports

All generated by `loci-cli eval` into `docs/reports/`. Speed and memory on this Mac are in `docs/BENCHMARKS.md` (generated by `loci-cli bench`).
