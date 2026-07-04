# Phase 3: Quality And Importance Diagnostics

## Goal

Make the importance model inspectable before it becomes powerful.

Phase 3 reports coverage, missingness, score distributions, and simple
correlations among available signals. It also runs policy sweeps so we can see
how much each retrieval policy shifts scores and source selection.

## Outputs

- `reports/quality_diagnostics_<mode>.json`
- per-signal summary statistics
- pairwise Pearson correlations
- top source and year distributions

Author score remains a separate signal because it can amplify prestige effects.
