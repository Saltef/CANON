# Phase 13: Labeling And Judgment

## Goal

Create the path from RAG outputs to human labels and lightweight automated
judgments.

## Implementation

The label task builder converts RAG eval outputs into annotation tasks. The
deterministic judge scores whether answers have citations, claim backing,
cluster diversity, and conflict notes.

The calibration layer now compares completed human labels against technical
proxy labels. This gives CANON a way to ask whether deterministic metrics are
actually tracking expert judgment, rather than silently treating the proxy as
truth. Empty label files still produce useful technical-label distributions, but
agreement rates remain zero until human labels are filled in.

Current human-label dimensions:

- citation support
- answer faithfulness
- importance fit
- paradigm coverage
- evidence role fit
- source trust fit
- safety handling
- corroboration handling

## Outputs

- `reports/label_tasks_<mode>.json`
- `reports/label_judgments_<mode>.json`
- `reports/label_calibration_<mode>.json`
