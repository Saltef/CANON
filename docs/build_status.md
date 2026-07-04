# Build Status

## Completed

- Dockerized Python ingestion service.
- Docker Compose Postgres + pgvector service.
- Optional GROBID service profile.
- Phase 0 domain decision and seed topic configuration.
- Seed gold queries.
- OpenAlex URL construction, harvesting hook, and fixture loader.
- OpenAlex work normalization.
- OA PDF URL extraction from OpenAlex locations.
- Abstract-based section-aware chunking scaffold.
- Source/text importance signal placeholders.
- Explicit author-score signal.
- Corpus diagnostics report.
- Postgres schema for works, chunks, and ingest runs.
- Unit tests for OpenAlex normalization, chunking, diagnostics, and quality signals.
- Phase 3 quality/importance diagnostics.
- Phase 4 seed-query retrieval evaluation.
- Phase 5 citation graph diagnostics.
- Phase 6 claim extraction and conflict-candidate scaffold.
- Phase 7 versioned claim model artifact with features, weights, thresholds,
  explanations, and model inspection reports.
- Phase 8 deterministic semantic retrieval baseline.
- Phase 9 grounded citation synthesis.
- Phase 10 end-to-end RAG evaluation.
- Phase 11 embedding provider interface and local embedding store.
- Phase 12 generation provider interface.
- Phase 13 labeling task and deterministic judgment harness.
- Phase 14 topic-pack corpus expansion scaffold.
- Phase 15 static local workbench.
- Phase 16 named corpus validation loop.
- Phase 17 technical-only method evaluation pipeline.
- Phase 18 adversarial technical probe suite.
- Support-aware abstention diagnostics.
- Query-aware conflict-note surfacing.
- Expanded method variants for conflict awareness, source quality, and recency.
- Larger batch-evaluation target.
- Embedding provider comparison runner.
- pgvector SQL load-plan generator.
- GROBID full-text processing plan.
- Static evaluation dashboard.
- Technical weight-tuning runner.
- Reproducible experiment manifest builder.
- BM25 retrieval with lexical, balanced, importance, recency, and diverse policies.
- Retrieval traces with base, diversity, final, and cluster scores.
- Diversity-first audit for separating useful breadth from noisy breadth on the
  10k corpus.

## Intentionally Deferred

- PDF download and GROBID full-text parsing at scale.
- External dense embedding providers.
- Production pgvector embedding index.
- Full LLM-backed answer synthesis evaluation.
- Supervised or LLM-backed claim model training over labeled scholarly claims.
