# Phase 0: Scope

## Decision

Start with Political Science / International Relations.

## Why This Domain

- Debates are visible and useful for testing disagreement-aware RAG.
- Citation and bibliographic metadata are usually available for journal articles.
- The domain lets us test whether citation communities recover topics, methods,
  theoretical traditions, or some mixture of all three.

## Reframed Thesis

CANON tests how different importance signals affect scholarly RAG. It does not
assume that "quality", "centrality", "recency", or "tradition coverage" are the
same thing.

## Confirmatory Contrast

The first confirmatory comparison is:

Baseline hybrid retrieval + rerank vs. hybrid retrieval + transparent importance
fusion.

Everything else starts exploratory until the gold set is frozen.

## Phase 0 Exit Criteria

- Domain selected.
- Seed OpenAlex concepts and topic terms versioned.
- Seed gold queries versioned.
- Importance signals named and configured.
- Phase 1 ingestion has deterministic fixture path.
