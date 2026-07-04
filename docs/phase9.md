# Phase 9: Grounded Synthesis

## Goal

Generate answers from retrieved evidence while preserving citations,
importance traces, extracted claims, and conflict notes.

## Implementation

The first synthesizer is deterministic and extractive. It composes a short answer
from retrieved chunks and top claim-model outputs. This gives us an answer schema
that can later be served by an LLM while preserving the same evidence contract.

## Outputs

- `reports/synthesis_<mode>_<policy>_<query>.json`
- cited evidence items with chunk IDs and ranks
- importance summary for the answer context
- conflict notes when retrieved claims overlap known tensions
