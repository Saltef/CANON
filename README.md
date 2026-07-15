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

Start the product API:

```powershell
python -m canon.product.server --host 127.0.0.1 --port 8000
```

Check the API:

```powershell
Invoke-WebRequest http://localhost:8000/health
Invoke-RestMethod -Method Post http://localhost:8000/v1/sources/profile -ContentType "application/json" -Body '{"input_path":"data/my_docs","sample_size":25}'
```

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

See [docs/evaluation.md](docs/evaluation.md) and
[docs/human_review_rubric.md](docs/human_review_rubric.md).

## Public Docs

- [Quickstart](docs/quickstart.md)
- [Drive-First RAG Workflow](docs/drive_first_rag.md)
- [Supported Sources](docs/supported_sources.md)
- [Evaluation and Human Review](docs/evaluation.md)
- [Human Review Rubric](docs/human_review_rubric.md)
- [Testing on Your Own Documents](docs/test_own_documents.md)
- [Use Case and Product Boundary](docs/use_case.md)
- [Scientific Defensibility](docs/scientific_defensibility.md)

Internal planning, phase notes, and development checklists live in
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
reports/     generated reports, gitignored
tests/       regression and product tests
```

## Status

CANON is usable for local corpus testing, evidence-packet experiments, and
human-reviewed retrieval evaluation. Treat it as a serious workbench under
active development, not as a finished autonomous research product.
