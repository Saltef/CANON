# Evaluation And Human Review

CANON separates automated triage from human-reviewed quality claims.

## Automated Pre-Human Checks

The pre-human gate can run without manual labels:

```powershell
python -m canon.product.prehuman_check --mode my_topic_v1_corpus --benchmark-id llm_judged_my_topic_v1 --judge-provider heuristic --model-providers local --rerankers heuristic --top-k 10 --candidate-k 25
```

The same gate is available through the product API:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/v1/prehuman-check -ContentType "application/json" -Body '{"mode":"my_topic_v1_corpus","benchmark_id":"llm_judged_my_topic_v1","judge_provider":"heuristic","model_providers":["local"],"rerankers":["heuristic"],"top_k":10,"candidate_k":25}'
```

This prepares qrels review candidates, fills provisional relevance labels, runs
semantic-model evaluation, rerank evaluation, source diversity, smoke checks,
and readiness checks.

The strongest acceptable status from this path is:

```text
automated_pass_human_review_required
```

Use this for packaging confidence and triage, not final model claims.

## Human-Reviewed Qrels

Prepare review candidates:

```powershell
python -m canon.eval.qrels_review prepare --mode my_topic_v1_corpus --top-k 10
```

Import reviewed labels:

```powershell
python -m canon.eval.qrels_review import-csv --csv reports/qrels_review_tasks_my_topic_v1_corpus.csv --benchmark-id my_topic_qrels --output gold/my_topic_qrels.json
```

Evaluate retrieval models against reviewed qrels:

```powershell
python -m canon.eval.model_evaluation --mode my_topic_v1_corpus --qrels gold/my_topic_qrels.json --providers local,openai,cohere --k 10
```

Evaluate rerankers:

```powershell
python -m canon.eval.rerank_evaluation --mode my_topic_v1_corpus --qrels gold/my_topic_qrels.json --rerankers heuristic,cohere --base-policy rag --candidate-k 25 --k 10
```

## Report-Level Review

Use the industry-pilot workflow when testing answer usefulness, citation
quality, unsupported claims, and reviewer acceptance:

```powershell
python -m canon.product.industry_pilot --mode my_topic_v1_corpus --prepare-review
python -m canon.product.industry_pilot --mode my_topic_v1_corpus --records reports/human_review_tasks_v1.json --export-review-csv --output reports/human_review_tasks_v1.review.csv
python -m canon.product.industry_pilot --mode my_topic_v1_corpus --records reports/human_review_tasks_v1.json --import-review-csv reports/human_review_tasks_v1.review.csv
python -m canon.product.industry_pilot --mode my_topic_v1_corpus --records reports/human_review_tasks_v1.json --review-status
```

Use [human_review_rubric.md](human_review_rubric.md) as the labeling standard.

## API Keys

Hosted providers are optional. Add keys only to `.env`; never commit them:

```powershell
Copy-Item .env.example .env
notepad .env
```

The local and heuristic paths are the safest first tests because they do not
send corpus content to external model APIs.
