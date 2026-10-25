# eval-small

A small evaluation dataset for `loci-cli eval`. All documents are original text written for this
repository about fictional places, people and studies (so no model can answer from memory) and are
released under the repository's MIT licence.

- `documents/` — 8 documents (Markdown, plain text, HTML and a 4-page PDF).
- `config.json` — chunk size for this dataset (64 tokens, 8 overlap): the documents are short, so the default 256 would leave too few chunks to tell the retrieval modes apart.
- `questions.jsonl` — questions in the format described in `docs/EVALUATION.md`: 41 answerable
  questions with ground-truth snippets (and pages for the PDF) and 8 unanswerable ones. About half are
  phrased with the document's own words; the rest are paraphrases.

`LociTests` validates that every relevant snippet occurs in its document (and on its page for PDFs).

`sources/` holds the text the PDF was rendered from; regenerate it with
`swift Scripts/make-eval-pdf.swift Tests/Fixtures/eval-small/sources/brindlemoor-trial-report.txt Tests/Fixtures/eval-small/documents/brindlemoor-trial-report.pdf`
(form feeds separate pages). It is not ingested.

## Support subset (`support.jsonl`)

93 hand-labelled (claim, passage) pairs for judging and calibrating answer verification. Each line:
`{"id", "claim", "passage", "document", "label", "split"}`. The passage is exact text from `document`; the claims were
written by hand, three per label per passage or so, and labelled by the rules below.

- `supported` — everything the claim says is stated in the passage or follows directly from it.
- `weak` — the claim is about what the passage says and partly supported, but adds a detail, cause or conclusion the passage does not give.
- `unsupported` — the passage does not back the claim: it contradicts it (a changed number, name or direction), concerns something the passage never mentions, or is unrelated.

The unsupported claims are deliberately hard: most contradict the passage by changing a figure or a name, so they stay on topic.
`split` is `calibration` or `test` (each label divided evenly): `loci-cli eval --calibrate` sets thresholds from the
calibration half only, and agreement is reported on both halves so a threshold fitted to one is checked on the other.
With so few items the numbers are noisy; treat differences of a few points as nothing.
