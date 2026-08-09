# Quickstart

This guide shows the shortest path from a folder of documents to a named CANON
corpus you can query and evaluate.

## 1. Run Tests

```powershell
python -m pytest
```

## 2. Run The Built-In Demo

```powershell
python -m canon.product.demo
```

The demo runs the AI infrastructure fixture end to end: ingest, grounded brief,
automated brief evaluation, alert digest, alert evaluation, review packet, and
handoff summary. A successful automated run ends with
`automated_pass_human_review_required`, because final quality still needs human
labels.

If CANON is installed as a package, the same command is available as:

```powershell
canon-demo
```

## 3. Prepare A Local Corpus Folder

Use a local folder, mounted Google Drive folder, or local git checkout.

Recommended first test:

```text
data/my_docs/
  memo.md
  sources.csv
  briefing.pdf
  notes.docx
```

Do not include secrets, private keys, `.env` files, credentials, or files you do
not want indexed into local artifacts.

## 4. Create A Project Boundary

```powershell
python -m canon.product.project --project-name "AI Infrastructure Geopolitical Risk" --domain "AI infrastructure and geopolitical risk" --regions "Latin America,Brazil,Chile,Mexico" --languages "English,Spanish,Portuguese" --issue-categories "energy demand,water and cooling,cloud dependency,sovereign AI" --report-types "weekly_intelligence_brief,regional_risk_report,alert_digest" --source-boundaries "G:\My Drive\CANON Corpus" --corpus-id ai_infra_geo_risk_corpus
```

This writes `reports/projects/<project_id>/project_config.json` and `.md`.
The config records the domain ontology, monitored regions/languages, issue
categories, desired reports, source/corpus boundary, and the rule that monitors
must not run without an explicit source boundary.

## 5. Profile The Source

```powershell
python -m canon.ingest.flexible --input data/my_docs --mode my_topic_v1 --profile-only
```

For a mounted folder:

```powershell
python -m canon.product.mounted_corpus --input "G:\My Drive\CANON Corpus" --mode my_topic_v1 --profile-only
```

For a local git checkout:

```powershell
python -m canon.product.mounted_corpus --input "C:\path\to\repo" --mode repo_review_v1 --profile-only
```

## 6. Ingest And Build A Corpus

```powershell
python -m canon.ingest.flexible --input data/my_docs --mode my_topic_v1
python -m canon.corpus.build --corpus-id my_topic_v1_corpus --from-modes my_topic_v1 --corpus-only
```

The mounted-corpus helper can do the same workflow:

```powershell
python -m canon.product.mounted_corpus --input "G:\My Drive\CANON Corpus" --mode my_topic_v1 --corpus-id my_topic_v1_corpus --domain my_domain
```

Generated raw/processed artifacts are written under `data/`. Generated reports
are written under `reports/`. Both locations are gitignored.

To refresh the hosted vector index after building the corpus, configure
`OPENROUTER_API_KEY`, `COHERE_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` in
`.env`, then run:

```powershell
python -m canon.embeddings.index --mode my_topic_v1_corpus --embedding-provider openrouter --embedding-model qwen/qwen3-embedding-8b --vector-backend qdrant
```

Qdrant is a replaceable ANN index, not the source of truth. Re-run the index
command after adding documents to the folder and rebuilding the corpus.
On a free Render web service, uploaded or generated corpus files are not durable
across restart, redeploy, or spin-down. Use the free target for workflow testing;
use a paid persistent disk or another durable corpus store before relying on
hosted user corpora.

The production API can run setup and write a local source manifest in one step:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/v1/production/corpus-setup -ContentType "application/json" -Body '{"input_path":"data/my_docs","mode":"my_topic_v1","corpus_id":"my_topic_v1_corpus","build_corpus":true,"index_vector_store":true,"vector_backend":"qdrant","index_embedding_provider":"openrouter","index_embedding_model":"qwen/qwen3-embedding-8b"}'
```

After you add, edit, or remove files, call refresh instead of blindly
rebuilding:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/v1/production/corpus-refresh -ContentType "application/json" -Body '{"input_path":"data/my_docs","mode":"my_topic_v1","corpus_id":"my_topic_v1_corpus","build_corpus":true,"index_vector_store":true,"vector_backend":"qdrant","index_embedding_provider":"openrouter","index_embedding_model":"qwen/qwen3-embedding-8b"}'
```

If the source manifest is unchanged, the refresh endpoint returns
`no_source_changes`. Use `"force": true` when files are unchanged but chunking,
embedding model, or vector-backend settings changed.

## 7. Query Through The API

Start the API:

```powershell
python -m pip install -e ".[serve,vectorstores,docs]"
python -m canon.product.asgi --host 127.0.0.1 --port 8000 --max-concurrency 8 --max-queue-depth 16
```

Health check:

```powershell
Invoke-WebRequest http://localhost:8000/health
```

See available routes and example request bodies:

```powershell
Invoke-RestMethod http://localhost:8000/v1/routes
```

Ask for evidence:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/v1/evidence-packets -ContentType "application/json" -Body '{"request_id":"req_001","project_id":"my_project","question":"What does this corpus say about grid risk?","mode":"my_topic_v1_corpus","evidence_requirements":{"top_k":10,"include_conflicts":true,"include_source_diversity":true,"include_query_diagnostics":true}}'
```

Run the production workbench with Qdrant-backed retrieval and typed model review:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/v1/production/evidence-workbench -ContentType "application/json" -Body '{"query":"What does this corpus say about grid risk?","mode":"my_topic_v1_corpus","retrieval_engine":"model_candidate_pool","candidate_scope":"vector_store","vector_backend":"qdrant","retrieval_provider":"openrouter","retrieval_model":"qwen/qwen3-embedding-8b","reranker_provider":"cohere","reranker_model":"rerank-v4.0-pro","generator_provider":"openrouter","generator_model":"openai/gpt-4.1-mini","run_model_review":true,"model_review_provider":"openrouter","model_review_model":"openai/gpt-4.1-mini","allow_external_model_review":true}'
```

Hosted retrieval sends query/chunk text to the selected embedding/rerank
providers. Hosted model review sends relevance-gated snippets to OpenRouter and
returns typed stance/extraction diagnostics; it is not a human-review label.

Check whether retrieved evidence visibly covers the frame you asked for:

```powershell
python -m canon.product.frame_coverage "What does this corpus say about grid risk?" --mode my_topic_v1_corpus --top-k 10
```

Frame coverage is diagnostic. It highlights missing dimensions and follow-up
queries, but human review is still required before treating coverage as complete.

For the intended Drive-first workflow, use the private corpus as the first
evidence source, then allow external expansion only when you want corroboration,
freshness checks, or gap filling. See [drive_first_rag.md](drive_first_rag.md).

## 8. Run The Pre-Human Gate

```powershell
python -m canon.product.prehuman_check --mode my_topic_v1_corpus --benchmark-id llm_judged_my_topic_v1 --judge-provider heuristic --model-providers local --rerankers heuristic --top-k 10 --candidate-k 25
```

This is an automated triage gate. It does not replace human qrels or final
answer review.

## 9. Check Product Readiness

```powershell
python -m canon.product.readiness --mode social_science_ir_v1_harvest10
python -m canon.product.final_check --mode ai_infra_geo_risk_demo --records reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.json --no-fail
```

`blocked_human_review` is an expected final-check status before human labels are
complete. It means the automated gates are separate from final acceptance.

## Troubleshooting

If the API returns `not_found`, the server is usually running but the path or
HTTP method is wrong. Check the route list:

```powershell
Invoke-RestMethod http://localhost:8000/v1/routes
```

The error response also includes `available_routes` for the method you used.
Most product actions are `POST` routes with JSON bodies; health and route
discovery are `GET` routes.
