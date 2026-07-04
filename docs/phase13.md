# Phase 13: Labeling And Judgment

## Goal

Create the path from RAG outputs to human labels and lightweight automated
judgments.

## Implementation

The label task builder converts RAG eval outputs into annotation tasks. The
deterministic judge scores whether answers have citations, claim backing,
cluster diversity, and conflict notes.

## Outputs

- `reports/label_tasks_<mode>.json`
- `reports/label_judgments_<mode>.json`
