# Phase 10: End-to-End RAG Evaluation

## Goal

Compare retrieval policies through the whole RAG path, not just rankings.

## Implementation

The RAG evaluator runs each seed query through retrieval and synthesis for each
policy. It records citation count, claim-backed citation count, cluster/source
diversity, conflict-note count, limitations, and policy-level averages.

## Outputs

- `reports/rag_eval_<mode>.json`
- per-query policy answer summaries
- policy-level RAG metrics
