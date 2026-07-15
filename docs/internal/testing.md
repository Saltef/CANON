# Internal Testing Cookbook

This file keeps the longer maintainer/testing commands out of the public README.

## Core Regression

```powershell
python -m pytest
```

Focused product/source tests:

```powershell
python -m pytest tests/test_flexible_ingest.py tests/test_mounted_corpus.py tests/test_product_server_routes.py tests/test_product_service.py
```

## Local API

```powershell
python -m canon.product.server --host 127.0.0.1 --port 8000
Invoke-WebRequest http://localhost:8000/health
Invoke-WebRequest http://localhost:8000/v1/summary
```

## Flexible Ingest

```powershell
python -m canon.ingest.flexible --input data/my_docs --mode my_topic_v1 --profile-only
python -m canon.ingest.flexible --input data/my_docs --mode my_topic_v1
python -m canon.corpus.build --corpus-id my_topic_v1_corpus --from-modes my_topic_v1 --corpus-only
```

Mounted corpus:

```powershell
python -m canon.product.mounted_corpus --input "G:\My Drive\CANON Corpus" --mode ai_infra_geo_risk_v1 --profile-only
python -m canon.product.mounted_corpus --input "G:\My Drive\CANON Corpus" --mode ai_infra_geo_risk_v1 --corpus-id ai_infra_geo_risk_v1_corpus --domain ai_infrastructure_geopolitical_risk
```

Git repo corpus:

```powershell
python -m canon.product.mounted_corpus --input "C:\path\to\repo" --mode repo_review_v1 --profile-only
```

## Pre-Human Automated Gate

```powershell
python -m canon.product.prehuman_check --mode my_topic_v1_corpus --benchmark-id llm_judged_my_topic_v1 --judge-provider heuristic --model-providers local --rerankers heuristic --top-k 10 --candidate-k 25
```

## Human Qrels Retest

```powershell
python -m canon.eval.qrels_review prepare --mode my_topic_v1_corpus --top-k 10
python -m canon.eval.qrels_review import-csv --csv reports/qrels_review_tasks_my_topic_v1_corpus.csv --benchmark-id my_topic_qrels --output gold/my_topic_qrels.json
python -m canon.eval.model_evaluation --mode my_topic_v1_corpus --qrels gold/my_topic_qrels.json --providers local,openai,cohere --k 10
python -m canon.eval.rerank_evaluation --mode my_topic_v1_corpus --qrels gold/my_topic_qrels.json --rerankers heuristic,cohere --base-policy rag --candidate-k 25 --k 10
```

## Industry Pilot Review

```powershell
python -m canon.product.industry_pilot --mode my_topic_v1_corpus --prepare-review
python -m canon.product.industry_pilot --mode my_topic_v1_corpus --records reports/human_review_tasks_v1.json --export-review-csv --output reports/human_review_tasks_v1.review.csv
python -m canon.product.industry_pilot --mode my_topic_v1_corpus --records reports/human_review_tasks_v1.json --import-review-csv reports/human_review_tasks_v1.review.csv
python -m canon.product.industry_pilot --mode my_topic_v1_corpus --records reports/human_review_tasks_v1.json --review-status
python -m canon.product.final_check --mode my_topic_v1_corpus --records reports/human_review_tasks_v1.json --no-fail
```

## Docker Path

```powershell
docker compose build canon
docker compose up canon
docker compose run --rm canon python -m pytest
```

The Docker service exposes the product API on `http://localhost:8000`.
