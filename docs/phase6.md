# Phase 6: Claim And Conflict Scaffold

## Goal

Introduce claim-aware retrieval diagnostics without requiring an LLM extractor yet.

Phase 6 extracts lightweight, deterministic claim candidates from retrieved
chunks and compares them for conflict-like tension. This is a scaffold for later
LLM-backed claim extraction, NLI checks, and typed disagreement surfacing.

## Claim Contract

Each claim candidate records:

- claim id
- work id and chunk id
- title, source, year, cluster id
- text span
- topic tokens
- stance
- confidence

## Conflict Candidate Contract

Each conflict candidate records:

- pair id
- claim ids
- shared topic tokens
- conflict type
- reason
- score

The current detector is intentionally conservative and heuristic. It should be
treated as a triage layer, not a truth oracle.
