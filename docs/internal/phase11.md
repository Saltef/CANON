# Phase 11: Embedding Providers

## Goal

Move from a single built-in semantic score toward swappable embedding providers.

## Implementation

The embedding layer supports a local deterministic provider, plus OpenAI and
Cohere API providers behind environment-gated call paths. Embeddings are written
as JSONL records keyed by chunk ID so they can later be moved into pgvector.

## Outputs

- `data/processed/embeddings_<mode>_<provider>.jsonl`
- `reports/embeddings_<mode>_<provider>.json`
