# CANON

CANON is a human-in-the-loop evidence briefing workbench for building cited,
cautious answers from controlled document corpora.

It helps a reviewer profile sources, ingest a corpus, compare retrieval methods,
generate evidence packets, and inspect where support is weak, missing, or
contested. CANON is not a truth oracle and does not replace expert review.

## What It Does

- Ingests local folders, mounted Google Drive folders, and local git checkouts.
- Searches a user's private corpus first, then supports controlled expansion to
  external sources for corroboration, freshness, and missing coverage.
- Supports text files, Markdown, CSV/JSON/JSONL, PDF, DOCX, XLSX, PPTX, HTML,
  notebooks, source code, and common config files.
- Detects Google-native pointers, images, and legacy presentations without
  pretending they are extractable evidence text.
- Builds named corpora with chunking, source metadata, and audit artifacts.
- Produces cited answers and evidence-packet JSON for downstream systems.
- Runs retrieval, rerank, source-diversity, qrels, and pre-human evaluation
  checks.
- Keeps automated judge output provisional until human review is completed.

## Human Review Boundary

CANON is designed for evidence triage and reviewable first drafts. Human review
is required before using outputs for final conclusions, publication, legal,
medical, financial, policy, or other high-stakes decisions.

## Quick Local Check

```powershell
python -m pip install -e .
make test
make ci
```

`make ci` runs install, unit tests, dry-run ingestion, report generation, and the product smoke/readiness gates with the same command path used in CI.

CANON's core package is intentionally stdlib-first. Hosted OpenRouter/Cohere
calls are made through HTTP APIs when keys are configured. Optional SDK/model
dependencies live under `pyproject.toml` extras; they are not required for the
local deterministic CI path.

The default `local` / `hashed-semantic-v1` encoder is an offline deterministic
fallback for tests and reproducible controls. It hashes lexical unigrams and
bigrams into a sparse vector. Treat it as a lexical/hash baseline, not as a
production neural embedding model.

Human review is required for final conclusions, citation validity, domain
interpretation, corpus representativeness, and any high-stakes use.

Automated checks can say "ready for human review." They should not be used to
claim final model quality, factual correctness, or release-level unsupported
claim rates without reviewed qrels and reviewed answer/report labels.

## Quick Start

Run the test suite:

```powershell
python -m pytest
```

Run the built-in flagship demo:

```powershell
python -m canon.product.demo
```

The demo ingests the fixture corpus, runs the grounded intelligence brief,
alert digest, automated gates, and review-packet preparation, then prints the
next human-review action plus key artifact paths. Installed packages also
expose this as `canon-demo`.

Create a scoped intelligence project before connecting private sources:

```powershell
python -m canon.product.project --project-name "AI Infrastructure Geopolitical Risk" --domain "AI infrastructure and geopolitical risk" --regions "Latin America,Brazil,Chile,Mexico" --languages "English,Spanish,Portuguese" --issue-categories "energy demand,water and cooling,cloud dependency,sovereign AI" --report-types "weekly_intelligence_brief,regional_risk_report,alert_digest" --source-boundaries "G:\My Drive\CANON Corpus" --corpus-id ai_infra_geo_risk_corpus
```

This writes a local `project_config.json` with ontology, source plan, monitor
boundary, and human-review limits under `reports/projects/<project_id>/`.

Profile and ingest your own local corpus:

```powershell
python -m canon.ingest.flexible --input data/my_docs --mode my_topic_v1 --profile-only
python -m canon.ingest.flexible --input data/my_docs --mode my_topic_v1
python -m canon.corpus.build --corpus-id my_topic_v1_corpus --from-modes my_topic_v1 --corpus-only
```

Use a mounted Google Drive folder or local git repo as a corpus source:

```powershell
python -m canon.product.mounted_corpus --input "G:\My Drive\CANON Corpus" --mode ai_infra_geo_risk_v1 --profile-only
python -m canon.product.mounted_corpus --input "C:\path\to\repo" --mode repo_review_v1 --profile-only
```

Try the built-in AI infrastructure geopolitical risk fixture:

```powershell
python -m canon.ingest.unstructured --input data/fixtures/ai_infra_geo_risk_sample.jsonl --mode ai_infra_geo_risk_demo --chunk-tokens 80 --overlap-tokens 10
python -m canon.synthesis.answer "What are the emerging geopolitical risks around AI data center expansion in Latin America?" --mode ai_infra_geo_risk_demo --policy rag --top-k 5
python -m canon.product.frame_coverage "What are the emerging geopolitical risks around AI data center expansion in Latin America?" --mode ai_infra_geo_risk_demo --policy rag --top-k 8
python -m canon.intelligence.evidence_runner "What are the emerging geopolitical risks around AI data center expansion in Latin America?" --mode ai_infra_geo_risk_demo --policy rag
python -m canon.product.report_quality "What are the emerging geopolitical risks around AI data center expansion in Latin America?" --mode ai_infra_geo_risk_demo --policy rag
python -m canon.eval.intelligence_brief --mode ai_infra_geo_risk_demo --queries-path gold/ai_infra_geo_risk_seed_queries.json --policy rag
python -m canon.intelligence.alerts "What are the emerging geopolitical risks around AI data center expansion in Latin America?" --mode ai_infra_geo_risk_demo --policy rag
python -m canon.eval.alert_digest --mode ai_infra_geo_risk_demo --queries-path gold/ai_infra_geo_risk_seed_queries.json --policy rag
python -m canon.product.intelligence_review --prepare-review --mode ai_infra_geo_risk_demo --queries-path gold/ai_infra_geo_risk_seed_queries.json --policy rag
python -m canon.product.flagship_handoff --mode ai_infra_geo_risk_demo
python -m canon.product.acceptance_scenario --mode ai_infra_geo_risk_demo
```

Start the product API:

```powershell
python -m canon.product.server --host 127.0.0.1 --port 8000
```

For the production serving path, install the optional ASGI stack and run the
FastAPI server with bounded concurrency:

```powershell
python -m pip install -e ".[serve]"
python -m canon.product.asgi --host 127.0.0.1 --port 8000 --max-concurrency 8 --max-queue-depth 16
```

`serve` keeps the core CANON package dependency-free while adding FastAPI,
uvicorn, pooled `httpx` provider calls, and Prometheus metrics. `otel` is a
separate optional extra for OpenTelemetry instrumentation:

```powershell
python -m pip install -e ".[serve,otel]"
```

The ASGI server writes structured JSONL operational logs to
`reports/asgi_operational_v1.jsonl`, carries a `request_id` through every log
line, emits stage spans/logs for `bm25`, `embed`, `fuse`, `rerank`, and
`synthesise`, exposes `/metrics`, rejects excess queue depth with HTTP 503 plus
`Retry-After`, and degrades embed/rerank failures to marked BM25/RRF fallbacks
instead of silently failing the whole request.

Open the local Evidence Discovery Workbench:

```text
http://127.0.0.1:8000/app
```

The workbench runs local corpus retrieval, evidence cards, query diagnostics,
coverage gaps, a cited draft preview, run diagnosis, and product feedback
capture. The run diagnosis explains corpus fit, candidate retrieval, reranking,
generation grounding, coverage gaps, and the remaining human-review boundary. By
default it does not call hosted models or external web search. When the OpenAlex
online search option is enabled, CANON sends only the query string to OpenAlex
and marks returned evidence as `ONLINE` / `external_source` so it cannot be
confused with corpus evidence. Feedback is stored as local user experience
telemetry, not as formal human-review labels.

Use the Corpus Setup panel in the app to profile, ingest, and build a corpus
from a local folder or file path. After setup, select the new mode in the Corpus
dropdown and run a question. The production workbench also runs a relevance gate:
if the selected corpus does not visibly match the query, it blocks the evidence
note and marks retrieved rows as diagnostic candidates instead of usable
supporting evidence.

Check the API:

```powershell
Invoke-WebRequest http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/v1/production/status
Invoke-RestMethod -Method Post http://localhost:8000/v1/production/corpus-setup -ContentType "application/json" -Body '{"input_path":"data/my_docs","mode":"my_topic_v1","corpus_id":"my_topic_v1_corpus","build_corpus":true}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/production/evidence-workbench -ContentType "application/json" -Body '{"query":"What are the grid risks around AI data center expansion?","mode":"ai_infra_geo_risk_demo","top_k":12,"freedom_level":"balanced","suggest_external_expansion":true}'
Invoke-RestMethod http://localhost:8000/v1/routes
Invoke-RestMethod -Method Post http://localhost:8000/v1/projects/start -ContentType "application/json" -Body '{"project_name":"AI Infrastructure Geopolitical Risk","domain":"AI infrastructure and geopolitical risk","regions":["Latin America","Brazil","Chile","Mexico"],"languages":["English","Spanish","Portuguese"],"issue_categories":["energy demand","water and cooling","cloud dependency"],"desired_report_types":["weekly_intelligence_brief","alert_digest"],"source_boundaries":["G:/My Drive/CANON Corpus"]}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/sources/profile -ContentType "application/json" -Body '{"input_path":"data/my_docs","sample_size":25}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/frame-coverage -ContentType "application/json" -Body '{"question":"What does this corpus cover about grid risk?","mode":"my_topic_v1_corpus","research_frame":{"subdomains":["energy","water"],"regions":["Latin America"],"languages":["English","Spanish"]},"evidence_requirements":{"top_k":10,"minimum_source_types":["official","local_media"]}}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/report-quality -ContentType "application/json" -Body '{"query":"What are the emerging geopolitical risks around AI data center expansion in Latin America?","mode":"ai_infra_geo_risk_demo","write_report":true}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/prehuman-check -ContentType "application/json" -Body '{"mode":"my_topic_v1_corpus","benchmark_id":"llm_judged_my_topic_v1","judge_provider":"heuristic","model_providers":["local"],"rerankers":["heuristic"],"top_k":10,"candidate_k":25}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/acceptance-scenario -ContentType "application/json" -Body '{"mode":"ai_infra_geo_risk_demo","write_report":true}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/intelligence-review/handoff -ContentType "application/json" -Body '{"records_path":"reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.json"}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/intelligence-review/feedback -ContentType "application/json" -Body '{"records_path":"reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.completed.json"}'
```

Run the local ASGI load ramp against a frozen query set:

```powershell
python -m scripts.load_test_asgi --base-url http://127.0.0.1:8000 --duration-per-level 200
```

The default ramp is 1, 5, 10, 20, 35, and 50 concurrent workers, for 20 minutes
total. It writes raw samples plus JSON/Markdown reports under `reports/load/`.
The first published local ASGI run is summarized in
[docs/asgi_load_test_report.md](docs/asgi_load_test_report.md).

Run the LlamaIndex framework baseline against the same Stage 1 fixed-qrels
protocol:

```powershell
python -m pip install -e ".[baselines]"
python -m canon.baselines.llamaindex_baseline --repeats 3 --no-resume
```

The current head-to-head is summarized in
[docs/llamaindex_head_to_head.md](docs/llamaindex_head_to_head.md), with the
compact machine-readable artifact in
`reports/llamaindex_stage1_head_to_head.json`.

More examples are in [docs/quickstart.md](docs/quickstart.md).

## Source Safety

CANON reads files you point it at. Do not ingest folders containing secrets such
as `.env` files, private keys, tokens, credentials, private customer data, or
other material you do not want indexed into local artifacts.

For local git repositories, CANON skips `.git`, dependency folders, build
outputs, caches, and common virtual environment folders by default.

API keys are optional. Keep them in `.env`, which is gitignored:

```powershell
Copy-Item .env.example .env
notepad .env
```

Never paste API keys into prompts, reports, docs, tests, or committed files.

See [SECURITY.md](SECURITY.md) for the current security policy and private
corpus handling guidance.

## Evaluation

CANON includes two evaluation tracks:

- Pre-human checks: qrels candidate generation, heuristic or LLM judge
  suggestions, semantic model evaluation, rerank evaluation, source diversity,
  smoke checks, and readiness checks.
- Human-reviewed checks: reviewed qrels and reviewed answer/report labels used
  to make stronger claims about model quality and release readiness.

Run the automated pre-human gate:

```powershell
python -m canon.product.prehuman_check --mode my_topic_v1_corpus --benchmark-id llm_judged_my_topic_v1 --judge-provider heuristic --model-providers local --rerankers heuristic --top-k 10 --candidate-k 25
```

Check package/product readiness:

```powershell
python -m canon.product.readiness --mode social_science_ir_v1_harvest10
python -m canon.product.final_check --mode ai_infra_geo_risk_demo --records reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.json --no-fail
```

Installed packages expose these as `canon-readiness` and
`canon-product-final-check`.

See [docs/evaluation.md](docs/evaluation.md) and
[docs/human_review_rubric.md](docs/human_review_rubric.md).

Current committed benchmark artifacts are deliberately compact:

- `gold/beir_scifact_full_qrels.json`
- `gold/beir_nfcorpus_full_qrels.json`
- `reports/stage1_fixed_qrels_v2_summary.json`
- `reports/stage1_fixed_qrels_v2_summary.md`
- `reports/stage1_fixed_qrels_v2_repeat_spread.json`
- `reports/stage1_fixed_qrels_v2_repeat_spread.md`
- `reports/stage1_payload_shift_probe.json`

The Stage 1 summary is a 30-query NFCorpus pilot under a fixed parent-qrels
protocol. It is not leaderboard-comparable to published BEIR results.

## Public Docs

- [Quickstart](docs/quickstart.md)
- [Drive-First RAG Workflow](docs/drive_first_rag.md)
- [Supported Sources](docs/supported_sources.md)
- [Evaluation and Human Review](docs/evaluation.md)
- [Evidence-Grounded Intelligence Runner](docs/intelligence_runner.md)
- [Human Review Rubric](docs/human_review_rubric.md)
- [Publishable Evaluation Package](docs/publishable_evaluation_package.md)
- [Testing on Your Own Documents](docs/test_own_documents.md)
- [Use Case and Product Boundary](docs/use_case.md)
- [Scientific Defensibility](docs/scientific_defensibility.md)

Internal planning, roadmaps, and development checklists live in
[docs/internal](docs/internal/README.md). They are kept separate from the public
presentation so the repo stays readable while still preserving the working
roadmap.

## Repository Layout

```text
canon/       core ingestion, retrieval, evaluation, product API
conf/        local settings and corpus-source configuration
data/        local raw/processed corpus artifacts, gitignored
docs/        public documentation
docs/internal/
             internal planning and testing notes
gold/        checked-in benchmark/qrels fixtures
reports/     generated reports, gitignored except curated public summaries
tests/       regression and product tests
```

## Status

CANON is usable for local corpus testing, evidence-packet experiments, and
human-reviewed retrieval evaluation. Treat it as a serious workbench under
active development, not as a finished autonomous research product.

CANON is released under the MIT License. Public release claims are still gated
by the automated release audit and human-reviewed evaluation requirements; the
license only clears repository reuse hygiene, not model-quality or factual
correctness claims.
