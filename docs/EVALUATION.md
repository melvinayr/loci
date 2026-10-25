# Evaluation

Loci's claim is that answers are grounded and citations are correct. This document defines how that claim is measured. Every number in the README or `docs/BENCHMARKS.md` must come from this harness.

## Dataset format

A dataset is a folder:

```
my-eval/
├── documents/            # pdf, md, html, txt
├── questions.jsonl
├── support.jsonl         # optional: hand-labelled (claim, passage) pairs for judging verification
└── config.json           # optional: {"chunkTokens": 64, "overlapTokens": 8, "chunker": "structural"}
```

Each line of `questions.jsonl`:

```json
{
  "id": "q17",
  "question": "What sample size did the 2024 trial use?",
  "relevant": [{"document": "trial-2024.pdf", "page": 4}],
  "answer": "412 participants",
  "answerable": true
}
```

- `relevant` — ground-truth passages, by document (required) and optionally page or a text snippet that must be contained in the retrieved chunk.
- `answer` — reference answer (optional; used for answer-match metrics).
- `answerable: false` — the documents do not contain the answer; the correct behaviour is refusal.

### Support subset (`support.jsonl`)

Each line labels how well one passage supports one claim, the thing answer verification has to judge:

```json
{"id": "sup-lh1-4", "claim": "About 140 masons built the tower, and every one of them came from Gilsby.", "passage": "…exact text from the document…",
 "document": "cape-marrow-lighthouse.md", "label": "weak", "split": "calibration"}
```

`label` is `supported` (everything in the claim is stated or follows directly), `weak` (on topic and partly supported, but adds a detail, cause or conclusion the passage does not give) or `unsupported` (contradicted, never mentioned, or unrelated). `split` is `calibration` or `test`, each label divided evenly. Include unsupported claims that stay on topic, such as a changed number or name: unrelated claims are easy and say little.

## Metrics

### Retrieval
| Metric | Definition |
|--------|-----------|
| Recall@k | Fraction of questions where at least one relevant passage is in the top k (k = 1, 5, 10) |
| MRR | Mean reciprocal rank of the first relevant passage |
| nDCG@10 | Graded ranking quality |

Reported per mode (keyword, vector, hybrid, `hybrid+rerank`) so the value of each stage is visible. `hybrid+rerank` appears when the knowledge base has a reranker (`--reranker`), and uses the top 20 fused candidates.

### Citations
| Metric | Definition |
|--------|-----------|
| Citation precision | Fraction of citations that point to a relevant passage, over answerable questions that were not refused |
| Citation recall | Fraction of answerable questions where at least one citation points to a relevant passage |
| Structural validity | Fraction of citations (including a refusal's listed sources) whose `range` slices to exactly the `excerpt` in the stored text and, for PDFs, lies inside its `page` (must be 100%) |

### Faithfulness
| Metric | Definition |
|--------|-----------|
| Supported fraction | Sentences classified `supported` / all sentences, over answers that were not refused |
| Unsupported rate | Sentences classified `unsupported`, same denominator |
| Verifier agreement | On `support.jsonl`: accuracy, and precision/recall/F1 of the `unsupported` flag and of `supported`, with a confusion matrix, for the calibration and test halves separately; plus the probability that a supported example outscores an unsupported one (AUC) |

### Refusal
| Metric | Definition |
|--------|-----------|
| Refusal accuracy | Correct refusals on `answerable: false` questions |
| False refusal rate | Refusals on answerable questions |

### Answer match (optional)
Normalised containment of the reference `answer` in the generated text (case and punctuation ignored). Simple and transparent; no LLM-as-judge in v1. A correct paraphrase counts as a miss, so read it as a lower bound.

### Performance
Retrieval time, rerank time, verification time, time to first token, total answer time, tokens/s, peak memory (the process peak so far), index size on disk, ingest throughput (chunks/s). Always reported with device model, OS version, model set and `PromptVersion`.

## Running

```bash
swift run loci-cli eval --dataset Tests/Fixtures/eval-small --models recommended --out report.md
swift run loci-cli eval --dataset ~/my-eval --models quality --modes all --json report.json
```

Sections: `--answers` (full answers; needs `--generator`), `--verifier` (agreement with `support.jsonl`), `--calibrate` (fit cut-offs; implies `--verifier`), `--min-precision 0.95`, `--no-retrieval`, `--thresholds supported,weak`. Other flags: `--reranker fixture|<path.gguf>` (adds the `hybrid+rerank` mode), `--gpu` (Metal instead of CPU), `--residency lowMemory|balanced|keepResident`, `--modes keyword,vector,hybrid,hybrid+rerank|all`, `--json report.json`, `--chunk-tokens`/`--overlap-tokens` (override the dataset's `config.json`), `--fail-on-regression baseline.json --margin 2` (exit code 2 if any mode's Recall@5 drops by more than the margin; CI uses `baseline/eval.json`, generated with `--models fixture --json baseline/eval.json`). Evaluation runs on CPU by default (repeatable) in a scratch knowledge base that is deleted afterwards. **Rerankers are LLM-sized, and in the pinned llama.cpp build the CPU path is very slow** (minutes for the 41-question fixture set), so run `hybrid+rerank` with `--gpu`; GPU and CPU results can differ slightly. CI therefore gates only the CPU modes (keyword, vector, hybrid) with the embedder-only fixture set.

```swift
let report = try await Evaluator(knowledgeBase: kb).run(dataset: .folder(url))
try report.write(to: url, format: .markdown)
```

## Calibration

`loci-cli eval --calibrate` scores the `calibration` half of `support.jsonl` and fits the cut-offs. `weak` flags as many unsupported claims as it can while at least `--min-precision` (default 95%) of the claims it flags really are unsupported; `supported` is fitted the same way for recognising supported claims, but only if the scorer separates `supported` from `weak` (AUC ≥ 0.7), otherwise it equals `weak` and no sentence is called weak. Ties take the middle of the tied range. Maximising F1 was the first objective and was rejected after the end-to-end check below. The fitted cut-offs are then reported on **both** halves, so the test half checks them on examples they were not fitted to. With `--answers` the answers run with the fitted cut-offs. Results are recorded per model set in `docs/MODELS.md`.

Calibration optimises agreement on labelled sentence–passage pairs, not the behaviour of whole answers, and the two diverge: sentences a real generator writes are compressed paraphrases that score lower than hand-written supported claims. Always run `--answers` with the fitted cut-offs and check the false refusal rate before adopting them. The reports in `docs/reports/` record this for the current model set.

## CI policy

- Every PR runs the small fixture dataset on CPU with deterministic settings.
- The PR fails if Recall@5 or citation precision fall by more than 2 percentage points versus `main`, or structural validity drops below 100%.
- Nightly runs the full dataset and publishes the report as a workflow artifact.

## Datasets

1. **Fixture set (in repo):** target ~20 permissively licensed documents and ~60 questions, including ~15 unanswerable ones and a hand-labelled support subset. Built by the maintainers. `Tests/Fixtures/eval-small` is the v0: 8 original documents about fictional subjects (so no model can answer from memory) and 49 questions, 8 of them unanswerable. It is deliberately small and easy to run; expect retrieval modes to be close to each other on it, and grow it before drawing conclusions about small differences. `LociTests` checks that every ground-truth snippet really occurs in its document and page.
2. **External benchmarks:** adapters for public retrieval datasets may be added; check each dataset's licence before including any data in the repo.
