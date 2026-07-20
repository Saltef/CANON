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

If provisional qrels cannot be written, the gate returns `qrels_review_required`
with the review CSV path and exact `import-csv` command to run after relevance
labels are completed. That status is a review handoff, not a Python failure.

The strongest acceptable status from this path is:

```text
automated_pass_human_review_required
```

Use this for packaging confidence and triage, not final model claims.

## Automated Multi-Topic Benchmark Suite

Run the Stage 1 suite across configured public/topic benchmarks:

```powershell
python -m canon.product.automated_benchmark_suite --suite conf/benchmark_suites/stage1_public_multi_topic.json
```

This produces aggregate topic summaries, semantic retrieval metrics, pooled
lexical/vector candidate-recall diagnostics, rerank metrics, and score
observability. The default suite includes `local` as the reproducible control
and OpenAI/Cohere as hosted model candidates. If API keys are missing, hosted
models are reported as unavailable so the model matrix is visibly incomplete
rather than silently treated as a loss.

It is meant for automated regression testing before human review; it does not
replace human qrels or brief-level acceptance labels.

Run the Stage 1 candidate-pool sweep after the model matrix is configured:

```powershell
python -m canon.product.stage1_sweep --suite conf/benchmark_suites/stage1_public_multi_topic.json --candidate-values 25,50,100
```

The default Stage 1 suite includes local control embeddings, Cohere Embed v4,
OpenRouter OpenAI text embeddings, OpenRouter Qwen3 Embedding 8B, and
OpenRouter BGE-M3. Qwen queries are instruction-prefixed according to its model
guidance; BGE-M3 and OpenAI-style embeddings use raw query/document text; Cohere
uses separate `search_query` and `search_document` input modes.

Run the AutoML-style Stage 1 optimizer when comparing full retrieval stacks:

```powershell
python -m canon.product.stage1_optimizer --suite conf/benchmark_suites/stage1_public_multi_topic.json --candidate-values 25,50,100
```

Every optimizer trial includes BM25 traditional text search as the sparse
candidate source, then varies the dense retriever, candidate pool size, fusion
method, reranker, and reranker document format. The optimizer caches completed
trials under `reports/stage1_optimizer_cache/`, so interrupted hosted-model
runs can resume without paying for completed trials again.

Useful focused run:

```powershell
python -m canon.product.stage1_optimizer --suite conf/benchmark_suites/stage1_public_multi_topic.json --dense-retrievers openrouter:baai/bge-m3,openrouter:qwen/qwen3-embedding-8b --rerankers cohere:rerank-v4.0-fast --candidate-values 25,50 --fusion-methods union,rrf,weighted_bm25_dense --document-format structured --objective-mode balanced --max-chunks-per-parent 1
```

The Stage 1 objective uses retrieval metrics (`candidate_recall`, `nDCG@k`,
`Recall@k`, `MRR@k`, `MAP@k`), score-gap observability, latency, and a relative
cost hint. Objective modes are `quality`, `balanced`, `low_latency`, and
`low_cost`. Text generation metrics such as BLEU, METEOR, ROUGE, and BERTScore
are reserved for answer/brief evaluation, where generated text can be compared
against reference answers.

Use `--max-chunks-per-parent` when qrels are document-level or when multiple
chunks from one parent source crowd out source breadth. A value of `1` enforces
one top-ranked chunk per parent before lower-ranked duplicate chunks are
appended, making top-k reports easier to review without changing first-stage
candidate recall.

Check whether automated Stage 1 completion criteria pass:

```powershell
python -m canon.product.stage1_gate --optimizer-report reports/stage1_optimizer_v1.json
```

During development, use `--no-fail` to write the report even when the gate is
blocked:

```powershell
python -m canon.product.stage1_gate --no-fail
```

## Public Corpus Acceptance Sets

SciFact 30-query import:

```powershell
python -m canon.ingest.beir --dataset-dir data/raw/external/scifact --mode beir_scifact_stage1 --benchmark-id beir_scifact_stage1_qrels --split test --max-queries 30 --include-qrels-documents --chunk-tokens 220 --overlap-tokens 0
```

NFCorpus 30-query import:

```powershell
python -m canon.ingest.beir --dataset-dir data/raw/external/nfcorpus --mode beir_nfcorpus_stage1 --benchmark-id beir_nfcorpus_stage1_qrels --split test --max-queries 30 --include-qrels-documents --chunk-tokens 220 --overlap-tokens 0
```

Then run the two-corpus Stage 1 acceptance suite:

```powershell
python -m canon.product.stage1_optimizer --suite conf/benchmark_suites/stage1_public_two_corpus_30.json --candidate-values 25,50 --fusion-methods union,rrf,weighted_bm25_dense --document-format structured --objective-mode balanced
python -m canon.product.stage1_gate --optimizer-report reports/stage1_optimizer_v1.json
```

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
