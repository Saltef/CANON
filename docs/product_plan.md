# CANON Product Plan

CANON is moving from a research harness toward an evidence workbench for teams
that need scholarly answers with visible claim boundaries.

## Product Shape

- Evidence answer API: grounded answers with citations, support level,
  limitations, and conflict notes.
- Retrieval comparison API: side-by-side policy and importance-signal behavior.
- Scientific guardrail API: audit status, claim-decision rules, data-card
  limitations, and regression-gate status.
- Portfolio demo path: Dockerized local API first, richer UI later.

## Non-Negotiable Guardrails

- The product must not claim a stable global winner while the claim-decision
  report blocks that claim.
- Internal qrels can support technical diagnostics, not public benchmark claims.
- Small-corpus limitations stay visible in product responses.
- Diagnostic baselines stay in the product only when labeled as diagnostic.
- The product corpus must be built from explicit source profiles. OpenAlex is
  the implemented primary scholarly index and author index; Crossref, Semantic
  Scholar, Unpaywall, and DOAJ/CORE remain planned verification/backfill sources
  until connectors are implemented and audited.

## First Product Endpoints

- `GET /health`
- `GET /v1/summary`
- `GET /v1/reports/audit`
- `GET /v1/reports/claim-decision`
- `GET /v1/reports/data-card`
- `GET /v1/reports/regression-gate`
- `GET /v1/reports/diversity`
- `GET /v1/diversity/queries`
- `GET /v1/diversity/queries/{query_id}`
- `POST /v1/answer`
- `POST /v1/compare`
- `POST /v1/diversity-audit`
