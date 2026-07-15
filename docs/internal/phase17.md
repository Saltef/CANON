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

## Pure Technical Calibration

CANON now has a technical-only calibration pass in
`canon/eval/technical_calibration.py`. It converts RAG outputs into proxy labels
for the same dimensions used by the human-labeling workflow:

- citation support
- answer faithfulness
- importance fit
- paradigm coverage
- evidence role fit
- source trust fit
- safety handling
- corroboration handling

This is not a replacement for expert review. It is a way to test whether
technical proxies behave sensibly before labels exist, and to identify where a
method is weak even when its overall retrieval score looks decent. For example,
the dry-run technical calibration currently surfaces weak paradigm and
corroboration coverage even when RAG improves over lexical retrieval.

## Transparent Calibration Model

`canon/modeling/calibration_model.py` fits a small ridge-regularized linear
model over the calibration rows. It can train on human labels when present, or
on weak technical targets when labels are absent. The report includes:

- learned feature weights
- top positive and negative features
- in-sample error
- leave-one-query-out error
- comparison against the unweighted proxy baseline

This model is intentionally simple. Its purpose is to test whether combining
signals improves calibration and to expose which signals are carrying the
prediction. If it cannot beat the proxy baseline under leave-one-query-out, that
is useful evidence against the current feature design.

## Pairwise Preference Model

`canon/modeling/preference_model.py` fits a transparent pairwise model over
within-query policy comparisons. This is a better technical fit for RAG ranking
than treating every method output as an isolated scalar regression row.

For each query, the model compares policies whose calibration targets differ by
at least a minimum margin. It creates both directions of the comparison:

- better policy over worse policy with target `1`
- worse policy over better policy with target `0`

The report includes:

- pairwise feature weights
- in-sample pairwise accuracy
- leave-one-query-out pairwise accuracy
- average margin and confidence
- per-policy target summaries
- target-source counts

This does not make the target itself correct. If the target is a technical
proxy, the model is only measuring whether the feature set can reproduce that
proxy ordering. Its scientific value comes from exposing ranking instability,
near ties, and features that fail under held-out queries. Human labels and
external anchors are still required before using the learned weights as a
quality model.

## Hard-Negative Anchor Preferences

`canon/eval/hard_negatives.py` defines a small deterministic anchor preference
set for known failure modes:

- prompt-injection exposure versus safety-gated retrieval
- dependent echo-count inflation versus independent corroboration
- source accessibility/source naming versus methodological evidence quality
- one-sided high-similarity evidence versus conflict-aware synthesis
- older semantic matches versus current authoritative evidence for market/legal
  style questions

These anchors are intentionally not broad expert labels. They are minimum
expectation tests. A model that cannot learn these preferences should not be
trusted for production ranking, but a model that passes them has only passed a
small regression/stress set.

The pairwise model can train on this source with:

`python -m canon.modeling.preference_model --source anchors --mode hard_negative_anchor_preferences_v1`

## Outputs

- `reports/method_eval_<mode>_<method_set_id>.json`
- `reports/evaluation_suite_<mode>_<method_set_id>.json`
- `reports/batch_eval_<mode>_<method_set_id>.json`
- `reports/stress_slices_<mode>_<method_set_id>.json`
- `reports/technical_calibration_<mode>.json`
- `reports/calibration_model_<mode>_technical.json`
- `reports/preference_model_<mode>_technical.json`
- `reports/hard_negative_anchor_preferences_v1.json`
- `reports/preference_model_hard_negative_anchor_preferences_v1_anchors.json`
