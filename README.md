# CANON

Importance-aware scholarly RAG for testing how source and text importance signals
change retrieval, citation selection, answer quality, and disagreement handling.

The current build covers Phases 0 through 21:

- Phase 0: domain scope, seed topics, seed gold queries, and build hypotheses.
- Phase 1: Dockerized ingestion scaffold for OpenAlex metadata, OA PDF resolution,
  section-aware chunking, quality/coverage diagnostics, and database schema.
- Phase 2: baseline retrieval policies with auditable importance traces.
- Phase 3: quality and importance diagnostics.
- Phase 4: seed-query retrieval evaluation harness.
- Phase 5: citation graph diagnostics.
- Phase 6: claim extraction and conflict candidates.
- Phase 7: explicit versioned claim model baseline.
- Phase 8: deterministic semantic retrieval baseline.
- Phase 9: grounded citation synthesis.
- Phase 10: end-to-end RAG evaluation.
- Phase 11: embedding provider interface and persistent embedding stores.
- Phase 12: generation provider interface for template/OpenAI synthesis.
- Phase 13: labeling task and judgment harness.
- Phase 14: multi-topic corpus pack manifests.
- Phase 15: static local workbench.
- Phase 16: named corpus validation loop.
- Phase 17: technical-only named method evaluation pipeline.
- Phase 18: adversarial technical probe suite for disagreement, relevance traps,
  diversity pressure, recency pressure, off-topic robustness, and method-focused
  retrieval.
- Phase 19: scientific defensibility dossier and machine-readable audit for
  portfolio/research presentation.
- Phase 20: deeper technical testing layers for qrels import/validation,
  bootstrap uncertainty, citation faithfulness, query perturbations, and
  regression gates.
- Phase 21: paired significance testing and corpus data-card provenance
  reporting.

## Quick Start

```powershell
docker compose build canon
docker compose up canon
docker compose run --rm canon python -m canon.ingest.pipeline --dry-run
docker compose run --rm canon python -m unittest discover -s tests
docker compose exec -T postgres psql -U canon -d canon -f /app/canon/db/migrations/0002_author_score.sql
docker compose exec -T postgres psql -U canon -d canon -f /app/data/processed/load_dry_run.sql
docker compose run --rm --no-deps canon python -m canon.retrieval.experiment "democratic peace conflict" --policy balanced
docker compose run --rm --no-deps canon python -m canon.retrieval.compare "democratic peace conflict"
docker compose run --rm --no-deps canon python -m canon.retrieval.compare "democratic peace conflict" --policies lexical,balanced,diverse
docker compose run --rm --no-deps canon python -m canon.retrieval.compare "democratic peace conflict" --policies lexical,semantic,rag
docker compose run --rm --no-deps canon python -m canon.quality.diagnostics --mode live
docker compose run --rm --no-deps canon python -m canon.eval.harness --mode live --top-k 5
docker compose run --rm --no-deps canon python -m canon.graph.build --mode live
docker compose run --rm --no-deps canon python -m canon.claims.extract --mode live
docker compose run --rm --no-deps canon python -m canon.claims.conflict --mode live
docker compose run --rm --no-deps canon python -m canon.claims.model --write-report "We find robust support for the democratic peace."
docker compose run --rm --no-deps canon python -m canon.synthesis.answer "What does the literature say about democratic peace?" --mode live --policy rag
docker compose run --rm --no-deps canon python -m canon.eval.rag --mode live --top-k 5 --policies lexical,balanced,semantic,rag
docker compose run --rm --no-deps canon python -m canon.embeddings.store --mode live --provider local
docker compose run --rm --no-deps canon python -m canon.corpus.packs --max-results 50
docker compose run --rm --no-deps canon python -m canon.corpus.expansion --mode social_science_ir_v1_harvest10 --target-work-count 10000
docker compose run --rm --no-deps canon python -m canon.workbench.build --mode live
docker compose run --rm --no-deps canon python -m canon.corpus.build --corpus-id social_science_ir_v1 --from-modes live --top-k 5
docker compose run --rm --no-deps canon python -m canon.eval.methods --mode social_science_ir_v1_harvest10
docker compose run --rm --no-deps canon python -m canon.eval.pipeline --mode social_science_ir_v1_harvest10
docker compose run --rm --no-deps canon python -m canon.eval.diversity --mode social_science_ir_10k
docker compose run --rm --no-deps canon python -m canon.eval.diversity_diagnostics --mode social_science_ir_10k
docker compose run --rm --no-deps canon python -m canon.eval.diversity --mode social_science_ir_10k --diverse-method-id focus_diverse_k5_template --baseline-method-id lexical_k5_template
docker compose run --rm --no-deps canon python -m canon.eval.diversity_diagnostics --mode social_science_ir_10k --diverse-method-id focus_diverse_k5_template --baseline-method-id lexical_k5_template
docker compose run --rm --no-deps canon python -m canon.eval.diversity_gate --mode social_science_ir_10k
docker compose run --rm --no-deps canon python -m canon.eval.batches --mode social_science_ir_v1_harvest10 --batch-sizes 1,3,5
docker compose run --rm --no-deps canon python -m canon.eval.slices --mode social_science_ir_v1_harvest10 --method-ids diverse_k5_template,rag_k5_template
docker compose run --rm --no-deps canon python -m canon.eval.probes --mode social_science_ir_v1_harvest10 --method-ids diverse_k5_template,rag_k5_template
docker compose run --rm --no-deps canon python -m canon.eval.batches --mode social_science_ir_v1_harvest10 --batch-sizes 5,10,25,50
docker compose run --rm --no-deps canon python -m canon.eval.qrels --input /app/gold/ir_qrels_social_science_ir_v1_harvest10.json --format canon
docker compose run --rm --no-deps canon python -m canon.eval.qrels --input /app/gold/public_qrels_beir_scifact_smoke.json --format canon
docker compose run --rm --no-deps canon python -m canon.eval.external_ir --mode social_science_ir_v1_harvest10 --k 10
docker compose run --rm --no-deps canon python -m canon.eval.uncertainty --mode social_science_ir_v1_harvest10 --metric nDCG@10 --samples 500
docker compose run --rm --no-deps canon python -m canon.eval.significance --mode social_science_ir_v1_harvest10 --metric nDCG@10 --samples 1000
docker compose run --rm --no-deps canon python -m canon.eval.faithfulness --mode social_science_ir_v1_harvest10 --query-limit 5
docker compose run --rm --no-deps canon python -m canon.eval.perturbations --mode social_science_ir_v1_harvest10 --query-limit 8
docker compose run --rm --no-deps canon python -m canon.reports.data_card --mode social_science_ir_v1_harvest10
docker compose run --rm --no-deps canon python -m canon.eval.providers --mode social_science_ir_v1_harvest10 --providers local,openai,cohere
docker compose run --rm --no-deps canon python -m canon.embeddings.pgvector --mode social_science_ir_v1_harvest10 --provider local
docker compose run --rm --no-deps canon python -m canon.fulltext.grobid --mode social_science_ir_v1_harvest10
docker compose run --rm --no-deps canon python -m canon.eval.tuning --mode social_science_ir_v1_harvest10
docker compose run --rm --no-deps canon python -m canon.reports.dashboard --mode social_science_ir_v1_harvest10
docker compose run --rm --no-deps canon python -m canon.reports.claim_decision --mode social_science_ir_v1_harvest10
docker compose run --rm --no-deps canon python -m canon.eval.regression_gate --mode social_science_ir_v1_harvest10
docker compose run --rm --no-deps canon python -m canon.experiments.manifest --mode social_science_ir_v1_harvest10 --experiment-id social_science_ir_v1_harvest10_full
docker compose run --rm --no-deps canon python -m canon.reports.scientific_audit --mode social_science_ir_v1_harvest10
```

Product API diversity endpoints:

```powershell
Invoke-RestMethod http://localhost:8000/v1/reports/diversity?mode=social_science_ir_10k
Invoke-RestMethod http://localhost:8000/v1/reports/diversity-diagnostics?mode=social_science_ir_10k
Invoke-RestMethod http://localhost:8000/v1/reports/diversity-gate?mode=social_science_ir_10k
Invoke-RestMethod "http://localhost:8000/v1/reports/diversity-diagnostics?mode=social_science_ir_10k&diverse_method_id=focus_diverse_k5_template"
Invoke-RestMethod "http://localhost:8000/v1/diversity/queries?mode=social_science_ir_10k&verdict=useful_breadth"
Invoke-RestMethod http://localhost:8000/v1/diversity/queries/ir-off-topic-001?mode=social_science_ir_10k
Invoke-RestMethod -Method Post http://localhost:8000/v1/diversity-audit -ContentType "application/json" -Body '{"mode":"social_science_ir_10k"}'
```

The product API runs at `http://localhost:8000`:

```powershell
Invoke-WebRequest http://localhost:8000/health
Invoke-WebRequest http://localhost:8000/v1/summary
Invoke-RestMethod -Method Post http://localhost:8000/v1/answer -ContentType "application/json" -Body '{"query":"What does the literature say about democratic peace?","mode":"social_science_ir_v1_harvest10","policy":"rag","top_k":5}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/compare -ContentType "application/json" -Body '{"query":"democratic peace conflict","mode":"social_science_ir_v1_harvest10","policies":["lexical","rag","diverse"],"top_k":5}'
```

`/v1/answer` returns compact cited evidence with `explanation` metadata for
each result. `/v1/compare` returns the same retrieval-result explanations under
each policy run. Explanations report top weighted score contributors,
score-adjustments such as diversity bonuses, and reason tags such as
`high_lexical_relevance`, `author_score_signal`, and `cluster_diversity_bonus`.

The dry run uses `data/fixtures/openalex_sample.json` and writes diagnostics under
`reports/`. Live harvesting is available with:

```powershell
docker compose run --rm canon python -m canon.ingest.pipeline --live --max-results 50 --enrich-authors
```

Live harvesting requires outbound network access and should use a real contact
email in `conf/settings.toml`.

For the product corpus, use OpenAlex as the primary scholarly index and author
index, then grow toward a larger named corpus:

```powershell
docker compose run --rm --no-deps canon python -m canon.corpus.build --corpus-id social_science_ir_10k --harvest --max-results 2200 --corpus-only --top-k 5 --policies lexical,balanced,semantic,rag,diverse,conflict_aware
docker compose run --rm --no-deps canon python -m canon.corpus.expansion --mode social_science_ir_10k --target-work-count 10000
```

The expansion profile is versioned in `conf/corpus_sources.json`. Implemented
coverage uses OpenAlex Works and Authors for work metadata, citation/reference
signals, OA locations, and author-score inputs. Crossref, Semantic Scholar,
Unpaywall, and DOAJ/CORE are listed as planned cross-check/backfill sources so
they do not silently inflate current scientific claims.

## Thesis

The system does not assume a single definition of importance. It records separate
signals and makes their effects auditable:

- semantic relevance
- source quality
- citation centrality
- author prominence
- venue/publisher legitimacy
- open-access and open-science indicators
- section role and claim-bearing text
- cluster/topic diversity
- semantic similarity
- recency

The first product is an experiment harness, not a polished app: the same query
will be run across retrieval policies so we can see which importance signals help,
hurt, or merely shift citations.

## Docker Services

- `canon`: Python 3.12 ingestion/evaluation image.
- `postgres`: Postgres 16 with pgvector.
- `grobid`: optional PDF-to-TEI service for later full-text parsing.

## Phase Status

- Phase 0: implemented as versioned docs/config/gold query seeds.
- Phase 1: implemented as runnable ingestion scaffolding with deterministic
  fixtures and schema.
- Phase 2: baseline lexical retrieval and importance trace harness.
- Phase 3: quality and importance diagnostics.
- Phase 4: seed-query retrieval evaluation harness.
- Phase 5: citation graph diagnostics scaffold.
- Phase 6: deterministic claim and conflict scaffold.
- Phase 7: explicit versioned claim model baseline.
- Phase 8: deterministic semantic retrieval baseline.
- Phase 9: grounded citation synthesis.
- Phase 10: end-to-end RAG evaluation.
- Phase 11: embedding providers and stores.
- Phase 12: generation providers.
- Phase 13: dormant labeling and judgment utilities.
- Phase 14: topic packs for corpus expansion.
- Phase 15: static inspection workbench.
- Phase 16: named corpus validation loop.
- Phase 17: technical-only named method evaluation pipeline.
- Phase 18: adversarial technical probe suite.
- Phase 19: scientific defensibility audit and portfolio dossier.
- Phase 20: deep technical testing suite for uncertainty, faithfulness,
  perturbation robustness, external qrels ingestion, and regression gates.
- Phase 21: paired significance reports and corpus data cards.
- Current extension: support-aware abstention diagnostics, query-aware conflict
  notes, expanded method variants, larger batches, provider comparison reports,
  pgvector load plans, GROBID full-text plans, static dashboards, weight tuning,
  BEIR/TREC-style internal-qrels IR metrics, bootstrap uncertainty, citation
  faithfulness checks, perturbation robustness, paired significance checks,
  corpus data cards, scientific claim-decision reports, regression gates,
  reproducible experiment manifests, and scientific audit reports.
