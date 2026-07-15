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

## 4. Profile The Source

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

## 5. Ingest And Build A Corpus

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

## 6. Query Through The API

Start the API:

```powershell
python -m canon.product.server --host 127.0.0.1 --port 8000
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

For the intended Drive-first workflow, use the private corpus as the first
evidence source, then allow external expansion only when you want corroboration,
freshness checks, or gap filling. See [drive_first_rag.md](drive_first_rag.md).

## 7. Run The Pre-Human Gate

```powershell
python -m canon.product.prehuman_check --mode my_topic_v1_corpus --benchmark-id llm_judged_my_topic_v1 --judge-provider heuristic --model-providers local --rerankers heuristic --top-k 10 --candidate-k 25
```

This is an automated triage gate. It does not replace human qrels or final
answer review.
