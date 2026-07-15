# Phase 8: Semantic Retrieval

## Goal

Add a semantic retrieval signal without making the build dependent on external
embedding APIs.

## Implementation

`hashed-semantic-v1` encodes tokens and token bigrams into deterministic sparse
vectors. The score is not meant to be a final embedding model. It is a portable
semantic baseline that lets us test whether non-lexical matching changes the
importance-aware RAG stack.

## Outputs

- retrieval traces include `semantic_similarity`
- new retrieval policies: `semantic` and `rag`
- semantic scores are auditable and reproducible in Docker
