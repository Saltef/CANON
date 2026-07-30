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

Long public benchmark runs are resumable. The suite caches completed benchmark
reports under `reports/automated_benchmark_cache/` and per-query rerank work
under `reports/automated_benchmark_query_cache/`. If a full run is interrupted,
rerun the same command to continue from completed cache entries. Use
`--no-resume` only when intentionally rebuilding every cached query.

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

## Disagreement Preservation Benchmark

Stage 2 can be checked against a controlled disagreement fixture:

```powershell
python -m canon.eval.disagreement_fixture --output gold/disagreement_preservation_publishable.json
python -m canon.eval.disagreement_preservation --benchmark gold/disagreement_preservation_publishable.json
```

The report scores whether synthesis links support, contradiction,
qualification, and distractor evidence correctly. Metrics include link recall,
link precision, stance accuracy, contradiction recall, citation integrity,
unsupported-claim rate, and the aggregate disagreement-preservation score.

This benchmark is useful for regression and artifact freezing, but it remains
an automated pre-human signal. Human-reviewed synthesis labels are still
required before publication-quality claims.

## Publishable Human Review Scaffold

Prepare the retrieval-qrels and synthesis-label handoff before assigning human
review:

```powershell
python -m canon.product.human_review_scaffold --suite conf/benchmark_suites/stage1_public_full.json --disagreement-benchmark gold/disagreement_preservation_publishable.json
```

The scaffold writes:

- `reports/publishable_human_review_scaffold_v1.json`
- `reports/publishable_retrieval_review_scaffold.csv`
- `reports/publishable_synthesis_review_scaffold.csv`

It defines retrieval labels for relevance and evidence role, and synthesis
labels for citation validity, stance correctness, claim support, missed key
evidence, reviewer action, and rationale. This prepares the review work; it
does not mark the human-review gate complete. The publishable package also
checks that the scaffold meets its configured distinct retrieval-query and
synthesis-case targets, so pilot-sized handoff CSVs remain blocked even before
reviewer labels are considered. Completed label rows must include a stable
`reviewer_id` for auditability.

Validate completed labels and write the suite-specific publishable review
status:

```powershell
python -m canon.product.publishable_review --retrieval-csv reports/publishable_retrieval_review_scaffold.csv --synthesis-csv reports/publishable_synthesis_review_scaffold.csv
```

The package gate requires `reports/publishable_human_review_status_v1.json`, not
the older generic `reports/human_review_status_v1.json`, so a stale review
artifact cannot accidentally satisfy publishable claims.

Build the reviewer-facing benchmark card after the automated suite, optimizer,
disagreement benchmark, and review-status artifacts exist:

```powershell
python -m canon.product.publishable_benchmark_card --automated-suite-report reports/automated_benchmark_suite_stage1_public_full_v1.json --optimizer-report reports/stage1_optimizer_v1.json
```

The card summarizes candidate-recall failures, ranking failures,
score-observability signals, disagreement-preservation status, and blocked
claims. Use it beside the package manifest; do not treat it as human-reviewed
publication evidence.

## Public Corpus Acceptance Sets

Full SciFact and NFCorpus imports:

```powershell
python -m canon.ingest.beir --dataset-dir data/raw/external/scifact --mode beir_scifact_full --benchmark-id beir_scifact_full_qrels --split test --include-qrels-documents --chunk-tokens 220 --overlap-tokens 0
python -m canon.ingest.beir --dataset-dir data/raw/external/nfcorpus --mode beir_nfcorpus_full --benchmark-id beir_nfcorpus_full_qrels --split test --include-qrels-documents --chunk-tokens 220 --overlap-tokens 0
```

Use `conf/benchmark_suites/stage1_public_full.json` for the publishable package
coverage gate. The 30-query commands below remain useful for quick iteration and
debugging, but they are pilot slices.

Freeze the publishable workflow from existing full-suite artifacts:

```powershell
python -m canon.product.publishable_workflow --suite conf/benchmark_suites/stage1_public_full.json
```

Use `--run-benchmarks` only when you intentionally want to rerun the resumable
Stage 1 benchmark suite and optimizer as part of the workflow.

The package manifest is expected to hash the suite file, configured public qrels
files, the disagreement-preservation fixture, benchmark/optimizer/card/review
reports, and the retrieval/synthesis review CSV handoffs. If any of those inputs
are missing, the package remains blocked.

The optimizer report must also contain completed trials for every fusion method
declared in `conf/benchmark_suites/stage1_public_full.json`; configuration alone
does not satisfy the fusion-diagnostics gate.

The human-review scaffold must contain enough distinct retrieval queries and
synthesis cases to meet its configured targets. The publishable disagreement
fixture contains 30 controlled cases so the synthesis scaffold can meet the
default target while still remaining an automated pre-human signal.

After package generation, verify the frozen artifact manifest:

```powershell
python -m canon.product.publishable_verify --package reports/publishable_package_canon_publishable_evidence_workflow_v1.json
```

This checks artifact paths, byte counts, and SHA-256 hashes. It does not replace
human review or prove model superiority.

Create a reviewer handoff bundle from the verified manifest:

```powershell
python -m canon.product.publishable_export --package reports/publishable_package_canon_publishable_evidence_workflow_v1.json
```

The export writes a bundle directory and zip archive under `reports/`. It is
useful for sharing the frozen evidence packet; it is not a new quality gate.

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
