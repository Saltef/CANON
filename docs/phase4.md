# Phase 4: Evaluation Harness

## Goal

Run the seed gold queries across retrieval policies and produce a compact report.

This phase evaluates retrieval traces, not generated answers. It gives us early
feedback about policy behavior before we add embeddings and generation.

## Metrics

- top-k result counts
- distinct works
- distinct sources
- average score components
- rank overlap against the lexical baseline
- per-query policy comparison tables

The seed queries are not a frozen benchmark yet. They are a build-time smoke set.
