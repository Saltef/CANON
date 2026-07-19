# CANON Finish Evaluation Plan

CANON should be finished as a human-in-the-loop evidence briefing pilot, not as
an autonomous truth system. The release claim is narrow: given a controlled
corpus and a focused question, CANON can retrieve evidence, explain coverage,
draft a cautious cited brief, prepare review tasks, and block release claims
until human labels exist.

## Operating Loop

1. Define a scoped project with domain, regions, languages, issue categories,
   report types, and source boundaries.
2. Profile and ingest the explicit source set.
3. Build a corpus with source metadata, chunking, and audit artifacts.
4. Run evidence packets for each acceptance question.
5. Generate briefs, alert digests, query diagnostics, frame coverage, and
   external-expansion plans.
6. Export human review packets for evidence relevance, citation quality,
   unsupported claims, usefulness, and missing perspectives.
7. Import reviewed labels, rerun retrieval/report evaluations, and only then
   run the final release audit.

## Evaluation Ladder

### Level 0: Repository Integrity

- Command: `python -m pytest`
- Required result: all tests pass.
- Purpose: catches regressions in ingestion, retrieval, evidence contracts,
  product APIs, review packet handling, and deterministic demo workflows.

### Level 1: Automated Workflow Smoke

- Command: `python -m canon.product.demo`
- Required result: `automated_pass_human_review_required`.
- Purpose: proves the fixture workflow can ingest, brief, evaluate, generate
  review artifacts, and stop at the human-review boundary.

### Level 2: Pre-Human Retrieval Benchmark

- Command: `python -m canon.product.prehuman_check --mode <corpus> --judge-provider heuristic --model-providers local --rerankers heuristic`
- Required result: `automated_pass_human_review_required` or an explicit
  `qrels_review_required` handoff.
- Purpose: produces candidate qrels, provisional labels, semantic model
  comparisons, rerank comparisons, source-diversity checks, smoke checks, and
  readiness checks.
- Limitation: provisional qrels can prioritize review, but cannot support public
  model-quality claims.

### Level 3: Human-Reviewed Retrieval Benchmark

- Required dataset: at least 30 acceptance questions with reviewed qrels.
- Metrics: Recall@10, nDCG@10, MRR@10, source diversity, slice performance by
  document type/domain/language/region, and query-variant stability.
- Pass target: at least 80% of questions return three or more
  reviewer-rated relevant evidence items in the top 10.
- Purpose: validates that the retrieval stack works for the actual pilot task.

### Level 4: Human-Reviewed Brief Benchmark

- Required labels: citation support, unsupported claim count, usefulness,
  missing perspectives, review time, and reviewer accept/reject decision.
- Pass targets:
  - At least 90% of briefs include citations, support assessment, limitations,
    and conflict notes when applicable.
  - Fewer than 5% of answer claims are reviewer-rated unsupported.
  - At least 70% of first-pass briefs can be reviewed in 15 minutes or less.
- Purpose: validates that retrieved evidence becomes a usable review artifact.

### Level 5: Release Audit

- Command: `python -m canon.product.final_check --mode <corpus> --records <review-packet>`
- Required result for release: `pass`.
- Acceptable pre-release result: `blocked_human_review`.
- Purpose: verifies product smoke, readiness, human-review completeness,
  acceptance feedback, source-report integrity, and release audit status.

## Benchmark Design

The benchmark should be built from real pilot questions, not only synthetic
fixtures.

- Acceptance set: 30-50 focused questions covering known subdomains, regions,
  source types, and languages.
- Relevance labels: top candidate evidence from multiple retrieval policies,
  labeled as relevant, partially relevant, not relevant, or unsafe/unsupported.
- Report labels: one row per brief claim, with evidence IDs, support rating,
  missing-context notes, and accept/revise/reject decision.
- Slice coverage: official sources, local media, policy reports, company
  filings, public-opinion signals, English/Spanish/Portuguese, and each target
  region.
- Regression set: every reviewer-rejected unsupported claim becomes a future
  negative test.

## What Still Needs Work

- Replace fixture-only proof with a real mounted corpus pilot.
- Complete reviewed qrels for the acceptance set.
- Complete report-level human review labels.
- Add benchmark reports that compare current retrieval policy against a simple
  lexical baseline and any semantic/rerank providers used in production.
- Add review-time capture to the human review packet.
- Promote reviewer-rejected evidence and unsupported claims into regression
  tests.
- Keep generated build artifacts out of source control.

## Finish Definition

The project is finish-ready when:

- `python -m pytest` passes.
- `python -m canon.product.demo` returns
  `automated_pass_human_review_required`.
- A real pilot corpus has a 30-question human-reviewed acceptance set.
- Retrieval passes the 80% top-10 evidence target.
- Brief review passes citation, unsupported-claim, and review-time targets.
- `python -m canon.product.final_check` returns `pass` for the reviewed packet.

