# How verification works

What the support badges mean, and what they cannot tell you.

## The check

After the generator writes an answer, Loci splits it into sentences and resolves each sentence's `[n]` markers to the passages they cite. The reranker is asked, for each sentence and its cited passages, whether the passage *supports* the claim, and the answer is converted to a score between 0 and 1. The score maps to a ``SupportLevel``: supported, weak or unsupported, using thresholds in ``AnswerOptions/supportThresholds``. A sentence without a citation gets the best matching passage attached when one clearly supports it. If most sentences are unsupported, the answer is replaced by a refusal.

## What it catches

Claims unrelated to their passage score low and are flagged. On the repository's labelled evaluation set the reranker separates supported from unsupported claims with an AUC of about 0.87.

## What it does not catch

A claim that is on the same topic as its passage but contradicts or overstates it usually scores as supported: the checker is a relevance model at heart. About two thirds of the unsupported claims in the evaluation set were of this kind. The badge means "the cited passage is about this", not "this is true", so the citation, one tap from the passage, is the real check.

## Thresholds

The defaults, 0.6 and 0.35, are conservative: they flag little but almost never refuse a correct answer. Cut-offs fitted to maximise F1 flagged more, but caused false refusals of 73% of answerable questions end to end and were rejected. `docs/MODELS.md` has the figures; ``AnswerOptions`` lets you change them, and <doc:EvaluatingYourKnowledgeBase> shows how to measure the effect on your own documents.
