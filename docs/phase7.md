# Phase 7: Claim Model

## Goal

Turn claim extraction from hidden heuristics into an explicit, versioned model.

The first model is `claim-model-baseline-v1`, a transparent lexical baseline
with a JSON artifact. It is intentionally simple but real: it has features,
weights, thresholds, predictions, explanations, and a model card.

## Why This Shape

We do not yet have enough labeled scholarly claims to train a statistical model
honestly. A transparent baseline gives us a stable contract and an auditable
failure surface. Later, a trained classifier or LLM-backed extractor can replace
the artifact while preserving the same output schema.

## Outputs

- `data/processed/claims_<mode>.json`
- `reports/claim_model_<mode>.json`
- each claim includes `model_id`, `features`, and `explanation`
