# Retrieval Model Selection

CANON does not yet have enough evidence to claim which semantic model is best
for production retrieval.

The current implementation can compare providers with:

```powershell
python -m canon.eval.model_evaluation --mode my_topic_v1_corpus --qrels gold/my_topic_qrels.json --providers local,openrouter,cohere --k 10
```

The Stage 1 automated suite also treats hosted models as core benchmark
candidates:

```powershell
python -m canon.product.automated_benchmark_suite --suite conf/benchmark_suites/stage1_public_multi_topic.json
```

In that suite, `local` is the reproducible control, while OpenRouter/Cohere are
candidate production model routes. Missing API keys are reported as unavailable rows,
not as model failures.

That proves the evaluation path works, but it does not prove a global winner.
Model choice must be made on the target corpus, source shapes, languages, query
styles, latency budget, privacy requirements, and human relevance labels.

## Starting Position

Use this default stack until corpus-specific evidence says otherwise:

- **Retrieval baseline:** hybrid retrieval with lexical plus semantic signals.
- **Local deterministic fallback:** `local` / `hashed-semantic-v1`, a hashed
  lexical n-gram control rather than a neural semantic model.
- **Remote candidates:** OpenRouter and Cohere embedding providers when API keys
  are configured. Vendor-named model IDs are routed through OpenRouter, not
  vendor-specific API keys.
- **Future open-weight candidates:** multilingual retrieval models such as BGE-M3
  or multilingual E5, added behind the same provider interface.

The local baseline is useful for reproducibility and offline testing. It is not
expected to be the best production semantic model, and it should be described as
a hashed lexical fallback in public materials.

## Why There Is No Universal Best Model

BEIR showed that retrieval performance changes across domains and tasks, and
that BM25 remains a strong baseline while reranking and late-interaction methods
often improve zero-shot performance at higher cost.

Recent multilingual embedding work also reinforces that model rankings are
sensitive to language, task family, benchmark composition, and aggregation
method. A model that wins averaged multilingual benchmarks may not win a
specific retrieval workload.

So CANON should select models by measured corpus performance, not by vendor
claims or public leaderboard position alone.

## Candidate Classes

### Lexical Baseline

Keep BM25 or equivalent sparse retrieval in every benchmark. Dense retrieval can
miss exact names, acronyms, IDs, numbers, legal citations, and rare terms.

Use lexical as:

- a baseline
- a hybrid component
- a safety check for exact-match evidence

### General Dense Embeddings

Use closed or open dense embedding models for semantic paraphrase matching,
user-friendly wording, and concept-level retrieval.

Evaluate:

- domain terminology
- vague/beginner queries
- paraphrase recall
- long chunk sensitivity
- false semantic neighbors

### Multilingual Embeddings

Use multilingual candidates when the corpus or users cross language boundaries.

Evaluate:

- same-language retrieval
- English query to non-English evidence
- native query versus translated query
- per-language Recall@10 and nDCG@10

### Hybrid + Reranking

For production-quality evidence retrieval, the likely strongest architecture is
not one embedding model alone. It is usually:

```text
sparse retrieval + dense retrieval -> fusion -> reranker -> evidence packet
```

Reranking should be evaluated separately from first-stage retrieval because it
adds latency and cost.

CANON can evaluate rerankers on the first-stage candidate pool:

```powershell
python -m canon.eval.rerank_evaluation --mode <corpus_id> --qrels gold/<corpus>_qrels.json --rerankers heuristic,cohere --base-policy rag --candidate-k 25 --k 10
```

Use Cohere rerank only if it improves `nDCG@10` or `MRR@10` on your reviewed
qrels without hiding source-diversity, disagreement, or false-balance warnings.

## Required Test Sets

Create at least 50 labeled questions for a real decision. Prefer 100 or more
when choosing a model for production.

Minimum slices:

- 15 straightforward evidence questions
- 15 vague or beginner-worded questions
- 15 terminology-sensitive questions
- 10 exact-name, acronym, date, number, or citation questions
- 10 contradiction/gap questions
- 10 multilingual or cross-lingual questions if multilingual use is in scope

Each question needs relevant chunk IDs with graded relevance:

```json
{
  "id": "q01",
  "query": "What delays battery storage projects?",
  "relevant": {
    "chunk:abc": 3,
    "chunk:def": 2
  }
}
```

Use `3` for directly useful evidence, `2` for partially useful evidence, and
`1` for weak supporting context.

CANON can prepare the first human relevance-review packet by pooling candidates
from several retrieval policies:

```powershell
python -m canon.eval.qrels_review prepare --mode <corpus_id> --top-k 10
```

This writes:

- `reports/qrels_review_tasks_<corpus_id>.json`
- `reports/qrels_review_tasks_<corpus_id>.csv`
- `reports/qrels_review_tasks_<corpus_id>.md`

The reviewer fills the CSV `relevance` column using:

- `3`: directly answers or strongly supports the query
- `2`: partially useful evidence or important qualification
- `1`: background/context only
- `0`: not relevant, misleading, or too tangential
- blank: not yet reviewed

After review, convert the CSV into canonical qrels:

```powershell
python -m canon.eval.qrels_review import-csv --csv reports/qrels_review_tasks_<corpus_id>.csv --benchmark-id <benchmark_id> --output gold/<benchmark_id>.json
```

Then run model evaluation against the reviewed qrels.

To reduce manual review, run provisional judge labels first:

```powershell
python -m canon.eval.llm_judge qrels --csv reports/qrels_review_tasks_<corpus_id>.csv --output reports/qrels_review_tasks_<corpus_id>.judged.csv --provider heuristic
python -m canon.eval.llm_judge qrels --csv reports/qrels_review_tasks_<corpus_id>.csv --output reports/qrels_review_tasks_<corpus_id>.openrouter_judged.csv --provider openrouter --model openai/gpt-4.1-mini
```

Judge labels are triage labels. Keep human labels authoritative, and audit all
high-priority rows plus a random sample of the rest.

## Decision Metrics

Primary:

- `nDCG@10`: rewards highly relevant evidence appearing early.
- `Recall@10`: checks whether the model finds known useful evidence.
- `MRR@10`: checks whether the first useful evidence appears quickly.

Secondary:

- `Precision@10`
- coverage at 1, 3, and 10
- score margin
- query latency
- indexing latency
- storage size
- cost per 1,000 documents
- unavailable-provider rate

Qualitative/human:

- citation support
- unsupported claim rate
- query-lingo usefulness
- drift risk
- review time
- false semantic neighbor examples

## Selection Rule

A model can become the default semantic model only if:

- it beats the current semantic baseline on `nDCG@10`
- it does not reduce `Recall@10` by more than 5%
- it does not reduce exact-name/acronym performance versus lexical fallback
- it keeps query p95 latency within the product budget
- it passes human review on citation support and drift
- its advantage holds across at least three query slices

If no model clearly wins, keep hybrid retrieval as the default and expose the
model comparison report instead of declaring a winner.

## Current Evidence Status

Current smoke evidence:

- `local` / `hashed-semantic-v1` runs successfully as a deterministic hashed
  lexical fallback.
- OpenRouter and Cohere providers are marked `unavailable` when API keys are absent.
- The smoke qrels file proves the evaluation path works.

This is enough to validate the harness. It is not enough to pick the production
model.

## Next Step

To choose the best retrieval model, create a labeled qrels file for the target
corpus and run:

```powershell
python -m canon.eval.model_evaluation --mode <corpus_id> --qrels gold/<corpus>_qrels.json --providers local,openrouter,cohere --k 10
```

Then compare:

- model leaderboard
- slice summaries
- top changed results
- unavailable providers
- recommendation confidence

The model decision should be written into the release notes only after this
report and human review agree.
