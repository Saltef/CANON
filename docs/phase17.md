# Phase 17: Technical Method Evaluation Pipeline

## Goal

Compare different RAG methods as named experimental treatments rather than
single policy flags.

This phase is technical-only. It does not generate or require human annotation
tasks.

## Method Contract

Each method specifies:

- retrieval policy
- top-k
- generator backend
- generator model

The default method set is `conf/methods/baseline_methods.json`.

## Metrics

The first evaluation pipeline computes:

- claim coverage
- cluster diversity
- source diversity
- context relevance
- semantic alignment
- citation support
- conflict awareness
- composite score
- pairwise retrieval overlap
- score margins
- near-tie detection
- metric spread
- retrieval stability
- dominance checks
- incremental batch stability
- stress-slice stability by query type and topic

These are intentionally transparent approximations of the dimensions emphasized
by RAGAS, ARES, OpenScholar, SQuAI, and TREC-style RAG evaluation: retrieved
context quality, answer grounding, answer relevance, citation support, and
evidence coverage.

## Outputs

- `reports/method_eval_<mode>_<method_set_id>.json`
- `reports/evaluation_suite_<mode>_<method_set_id>.json`
- `reports/batch_eval_<mode>_<method_set_id>.json`
- `reports/stress_slices_<mode>_<method_set_id>.json`
