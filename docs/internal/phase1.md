# Phase 1: Ingestion

## Goal

Build a Dockerized ingestion path that can:

1. Harvest scholarly metadata from OpenAlex.
2. Preserve raw responses.
3. Normalize works into stable internal records.
4. Resolve open-access PDF/full-text candidates where metadata permits.
5. Chunk abstracts and later full text section-aware.
6. Produce corpus diagnostics that expose data limitations.
7. Emit a deterministic SQL load file for Postgres.

## Diagnostics Treated As First-Class Output

- total harvested works
- usable abstracts
- open-access works
- PDF URL coverage
- reference-list coverage
- retraction-flag coverage
- publication-year distribution
- source/venue distribution

## Non-Goals In Phase 1

- Embedding generation.
- Retrieval.
- Generation.
- Claim extraction beyond deterministic section/abstract scaffolding.
- Assuming citation communities represent paradigms.
