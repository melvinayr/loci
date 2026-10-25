# Evaluating your knowledge base

Measure retrieval and answer quality on your own documents with `LociEval`.

## A dataset

A dataset is a folder with `documents/`, a `questions.jsonl` of questions with their expected source passages (and unanswerable ones), and an optional `config.json`. `Tests/Fixtures/eval-small` in the repository is a complete example.

## Run it

```bash
swift run loci-cli eval --dataset ./my-dataset --models recommended --answers --out report.md
```

The report lists Recall@k, MRR and nDCG for each retrieval mode, citation precision and recall, the share of supported sentences, refusal accuracy on unanswerable questions, the false-refusal rate, and timings. `--json` writes the same for machines, and `--fail-on-regression baseline.json` fails when a metric drops beyond the margin, which suits CI.

## Reading it

Citation structural validity must be 100%: every citation's range slices to exactly its excerpt. Refusal accuracy matters more than answer match, which is a plain containment check. With a few dozen questions a difference of a few points is noise; add questions before drawing conclusions.
