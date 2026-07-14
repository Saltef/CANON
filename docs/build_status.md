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
- Document-type-aware boundary chunking with deterministic token-window
  fallback and data-card diagnostics for chunk length, source position,
  boundary alignment, chunk strategy, chunk-policy document type, chunk
  resolution, parent-context mode, generation-context role, parent-expansion
  policy, evidence-containment score, and safety-contamination risk.
- Deterministic parent-child chunk metadata with `atomic_chunk_id`,
  `parent_context_id`, parent-context counts, expansion-eligible share, and
  context-expansion policy counts in data cards.
- Safety-gated parent context expansion for synthesis: retrieved atomic chunks
  can expand to sibling parent context only when the chunk policy allows it and
  the assembled parent text passes safety screening.
- Chunking policy now canonicalizes raw source-form aliases such as `blog` and
  `court_opinion` through the document-type profiler, while unknown named
  uploads remain `unknown_unstructured_text` instead of being promoted to
  academic evidence.
- Chunking now enforces strategy-specific boundaries: public social/forum text
  is split into atomic sentence-level evidence units with no overlap to isolate
  prompt-injection spans, while news/open-web text preserves paragraph
  boundaries before falling back to sentence windows.
- Web and unknown unstructured text now switch to atomic sentence chunks when
  instruction-like sentences are detected, preventing benign web claims from
  being merged with prompt-injection text before the safety gate runs.
- Chunking strategy benchmark for evidence-span containment, forbidden-merge
  avoidance, boundary alignment, parent-context policy, safety-contamination
  risk, and unsafe prompt-injection span isolation across heterogeneous
  academic, psychology, anthropology, history/OCR, legal, market, filing, news,
  web, social, and transcript fixture cases.
- Chunking variant benchmark,
  `reports/chunking_variant_eval_ct1000_ov100.json/.md`, compares fixed token
  windows, the current boundary-aware policy, atomic sentence chunks, a lexical
  semantic-cohesion surrogate, local hashed embedding-semantic grouping, and
  paragraph-or-sentence chunking on span preservation, forbidden-merge
  avoidance, unsafe-span isolation, fragmentation, balanced utility, and
  safety-first objectives.
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
- Corpus-profiled syntax/form relevance is now a first-class retrieval
  component: candidate retrieval blends lexical, hashed semantic, syntax, and
  focus signals; final policies can weight `syntax_relevance`; traces expose
  query syntax targets, document syntax vectors, primary syntax category, and
  corpus syntax rarity weights.
- Syntax/form retrieval ablation,
  `reports/syntax_retrieval_ablation_v1.json/.md`, compares no-syntax,
  candidate-only syntax, and candidate-plus-rerank syntax variants across
  method, limitation, filing/risk, market, transcript, archival, and
  public-expression queries. The current fixture is saturated, so syntax is
  neutral on expected-evidence MRR rather than proven broadly useful.
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
- Phase H mixed unstructured document-routing benchmark.
- Generic JSONL ingestion for heterogeneous unstructured records.
- Source-agnostic named corpus assembly for OpenAlex, JSONL, and neutral raw
  sources.
- Document-type slice diagnostics for candidate exposure, final exposure,
  ranking, and safety rejections across retrieval policies.
- Domain-fit profiles and domain-slice diagnostics for economics, psychology,
  anthropology, history, cultural studies, legal/market, and general academic
  unstructured corpora.
- Heterogeneous unstructured readiness gate combining contracts, slice
  diagnostics, data cards, safety checks, and explicit claim boundaries.
- Unstructured experiment coverage matrix that maps target domains, source
  forms, task families, label anchors, current fixture coverage, and expansion
  gaps before broad cross-domain claims are allowed.
- Expanded heterogeneous fixtures now include preprint, policy report, market
  report, social/forum public-opinion records, and an unknown unstructured-text
  negative control; the coverage matrix still blocks broad claims because
  source-form coverage is fixture-only and human/external labels are incomplete.
- Public social-media/forum ingestion lane for allowed public JSONL, JSON, CSV,
  and annotated HTML captures, with public-visibility filtering,
  privacy-preserving author hashing, public-opinion document/domain profiles,
  and safety-gate diagnostics before reranking.
- Deterministic public-opinion analysis report for platform/thread coverage,
  safety exposure, theme/stance probes, duplicate risk, author-hash activity,
  temporal coverage, and aggregation-readiness checks.
- Deterministic evidence committee diagnostics with separate relevance,
  source-trust, domain-fit, safety, corroboration, and public-opinion
  aggregation votes, plus final-context policy-violation reporting.
- `committee_rag` retrieval policy that turns the evidence committee from an
  audit-only diagnostic into an experimental hard gate before final scoring:
  only evidence allowed for `generator_context` can enter the selected final
  context, while aggregate-only, inspect-only, rejected, or quarantined
  candidates remain visible in rejected-candidate traces.
- Paired committee-gate policy comparison report,
  `reports/committee_gate_policy_comparison_v1.json/.md`, comparing audit-only
  `rag` against hard-gated `committee_rag` on selected-context policy
  violations, final-context retention, and abstention.
- Committee-gate usefulness audit,
  `reports/committee_gate_usefulness_audit_v1.json/.md`, which turns lost
  baseline-selected context and gated abstentions into human-label-ready review
  tasks with proxy triage for acceptable exclusion, review-needed loss, and
  potentially useful loss.
- Stratified unstructured human-label task pack,
  `reports/unstructured_human_label_tasks_v1.json/.md`, covering document type,
  evidence role, source trust, source independence, safety disposition, answer
  usefulness, and chunking quality across domain/document-type strata plus
  committee-gate loss, abstention, and chunking-variant review cases. The
  chunking-quality tasks ask reviewers to judge harmful merges, qualifier
  preservation, fragmentation, and preferred chunking variant. This is
  annotation readiness, not completed human validation.
- Unstructured label calibration now includes deterministic proxy labels for
  chunking harmful-merge risk, qualifier preservation, fragmentation, and
  preferred chunking variant, while keeping the report status at `needs_labels`
  until human labels exist.
- Evaluation anchor registry now includes a dedicated
  `human_chunking_quality_labels_v1` anchor and claim limit, so chunking
  quality cannot be treated as validated from deterministic fixtures alone.
- Committee-aware generation context gate that keeps aggregate-only,
  inspect-only, rejected, quarantined, and blocked evidence out of ordinary RAG
  synthesis context while retaining it for audit or aggregate analysis.
- Large-window chunking safety regression is now part of the unstructured
  portfolio contract, so the experiment must show that boundary-aware public
  expression chunks isolate injection text even when fixed windows would merge
  benign and unsafe sentences.
- Heterogeneous unstructured experiment portfolio that combines readiness,
  coverage matrix, phase gates, prompt-injection security, chunking,
  public-opinion aggregation, evidence committee, synthesis context-gating, and
  evaluation-anchor claim limits into
  `reports/unstructured_experiment_portfolio_v1.json/.md`.
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
- Live, compliant platform-specific social-media collectors and
  representativeness, bot, coordination, stance, and temporal-burst models for
  public-opinion analysis.
- Larger external unstructured corpora and expert/human labels for the
  unstructured portfolio beyond the current deterministic fixture bundle.
- Human/expert chunking-quality labels and external chunking benchmarks for
  harmful merge avoidance, qualifier preservation, fragmentation, and
  retrieval/generation usefulness.
