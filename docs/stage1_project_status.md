# CANON Stage 1 Project Status

Stage 1 is the evidence-discovery and retrieval-evaluation layer. Its job is to
prove that CANON can mount a corpus, retrieve plausible evidence, rank it,
explain failures, and produce a repeatable acceptance report before human review
and synthesis claims are trusted.

## Current Stage 1 Result

The current focused pilot uses:

- Corpus: `beir_nfcorpus_stage1_title_preserve`
- Acceptance set: 30 NFCorpus public BEIR queries
- Qrels: document-level BEIR labels transferred to chunks with fixed corpus-map
  expansion only when `parent_qrels=true`
- Sparse retrieval: BM25
- Dense retrieval: `openrouter:qwen/qwen3-embedding-8b`
- Fusion: `rrf`
- Reranker: `cohere:rerank-v4.0-fast`
- Candidate pool: 250 lexical plus 250 vector candidates
- Candidate document format: `structured`
- Parent cap: `max_chunks_per_parent=0` for unconstrained candidate coverage

This is a 30-query pilot slice, not the full 323-query NFCorpus BEIR test set.
It is useful for product debugging and regression testing, but it should not be
reported as a full public benchmark result or compared directly to published
full-corpus BEIR leaderboards.

### Fixed Parent-Qrels Protocol

Earlier status notes used a candidate-dependent `parent_qrels` implementation:
the evaluator promoted sibling chunks only when a system retrieved them. That
made the effective qrels a function of each run's candidate set, so different
retrieval configurations were scored against different labels. Those old
`parent_qrels` numbers are retired and must not be cited.

The fixed protocol is `fixed_corpus_parent_map_v2`. When `parent_qrels=true`,
CANON expands relevant chunks from the original qrels across the full
corpus-level chunk-to-parent map before reading any candidate output. Every
configuration is therefore scored against a fixed qrels set.

Fresh runs were executed on 2026-07-30 with a clean fixed-qrels cache at
`reports/rerank_query_cache_fixed_qrels_v2/`. The compact report is
`reports/stage1_fixed_qrels_v2_summary.json`.

| Configuration | Qrels scope | Candidate recall mean | Work-level hit rate mean | nDCG@10 | Recall@10 | MAP@10 | MRR@10 | Zero-hit queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen+Cohere | original chunk qrels | 0.365927 | 0.395415 | 0.393658 | 0.142102 | 0.294536 | 0.647222 | 4/30 |
| Auto query expansion + fixed parent qrels | full corpus parent map v2 | 0.382565 | 0.418675 | 0.395759 | 0.142102 | 0.296783 | 0.650000 | 4/30 |
| Auto query expansion + fixed parent qrels + parent-neighborhood candidates | full corpus parent map v2 | 0.416257 | 0.418675 | 0.393091 | 0.142102 | 0.293396 | 0.650000 | 4/30 |

The corrected single-run result is much less dramatic than the retired status
note. A follow-up repeat-spread check now runs each fixed-qrels configuration
three times with separate query-cache roots. The report is
`reports/stage1_fixed_qrels_v2_repeat_spread.json`.

| Configuration | Runs | nDCG@10 mean | nDCG spread pp | MAP@10 mean | Candidate recall mean |
|---|---:|---:|---:|---:|---:|
| Base Qwen+Cohere | 3 | 0.394511 | 0.141 | 0.295470 | 0.365927 |
| Auto query expansion + fixed parent qrels | 3 | 0.394649 | 0.022 | 0.295595 | 0.382565 |
| Auto query expansion + fixed parent qrels + parent-neighborhood candidates | 3 | 0.394767 | 0.282 | 0.295955 | 0.416257 |

Paired nDCG@10 deltas versus base change sign across repeats:
`-0.057pp` to `+0.105pp` for auto expansion, and `-0.101pp` to
`+0.136pp` for parent-neighborhood candidates. The current conclusion is
therefore: do not claim an auto-expansion top-10 ranking gain from this pilot.
Parent-neighborhood expansion still improves candidate coverage, but it should
not be presented as a top-k ranking improvement.

Benchmark integrity note: hand-written biomedical alias expansion is
quarantined behind `--benchmark-oracle-expansion`. Normal runs with
`--auto-query-expansion` use explicit query variants and corpus-derived topic
profiles only. Any run using `--benchmark-oracle-expansion` is diagnostic and
must be excluded from headline metrics.

The automated gate still reports `automated_fail` because broad public qrels
make full candidate recall unrealistic at this candidate budget. That is useful
signal, not a plumbing failure: four of the 30 pilot queries still have no
relevant evidence in the candidate pool, and no reranker can recover evidence
that was never retrieved.

## What Was Fixed

The rerank evaluator now records candidate-level reranker scores, allowing
oracle-style diagnostics over relevant and non-relevant score distributions.
This makes Cohere's relevance scores useful even when top-k ranking does not
improve every metric.

The rerank evaluator also supports parent-aware top-k diversification through
`--max-chunks-per-parent`. This prevents multiple chunks from the same parent
document from crowding out other sources in the final ranked evidence packet.

The Stage 1 optimizer passes the parent cap into every trial and includes it in
cache keys and reports, so experiments remain reproducible.

The retrieval layer now builds per-corpus topic profiles through
`canon.retrieval.topic_profile` / `canon-topic-profile`. These profiles capture
top corpus terms, phrases, topic buckets, and related terms, then feed
auto-query expansion in the candidate generator. This keeps expansion
reproducible while letting each mounted corpus contribute its own vocabulary.

The hand-authored biomedical alias table remains available only as a benchmark
oracle diagnostic through `--benchmark-oracle-expansion`. It is intentionally
off by default so that public-qrels labels do not leak into normal retrieval
claims.

Hybrid score fusion now normalizes BM25 and dense scores independently before
combining them in `weighted_bm25_dense`. This prevents raw BM25 magnitudes from
swamping dense cosine similarities and makes fusion sweeps more interpretable.

The candidate generator now supports parent-neighborhood expansion. When a
chunk from a parent work is retrieved, Stage 1 can add a bounded number of
sibling chunks from that work as candidate evidence. Under the fixed qrels
protocol this raised candidate recall from `0.382565` to `0.416257`, but it
did not improve top-10 ranking.

The production research workflow can optionally enrich the deterministic
research frame through OpenRouter. That gives Stage 1 a non-deterministic frame
construction path for topic, scope, concepts, alternate queries, and review
notes, while preserving deterministic fallbacks for repeatable local tests.

## Oracle Reranking Diagnosis

The previous heuristic reranker had weak discrimination:

- relevant score mean: `0.133563`
- non-relevant score mean: `0.093827`
- AUC-like separation: `0.556983`
- overlap rate: `0.886035`
- interpretation: `heavy_overlap_or_weak_discrimination`

Cohere Rerank gave much stronger observability in the earlier diagnostic run:

- relevant score mean: `0.346928`
- non-relevant score mean: `0.245519`
- AUC-like separation: `0.856008`
- overlap rate: `0.287985`
- interpretation: `strong_discrimination`

This remains a useful diagnostic hypothesis, but these score-distribution
figures should be treated as observability evidence rather than final benchmark
claims until regenerated under the same fixed-qrels protocol used above.

## Qrels Semantics

For the NFCorpus pilot, qrels are document-level labels transferred to chunks.
The fixed protocol expands those labels from the full corpus chunk-to-parent map
before scoring any retrieval output. The evaluation therefore reports both:

- chunk-level recall: whether relevant chunks were retrieved
- work-level recall: whether any chunk from a relevant parent document was
  retrieved

The rule is explicit: every chunk whose parent BEIR document is relevant is
treated as relevant before candidates are inspected. This is a document-level
diagnostic for public BEIR labels. It is not a substitute for human-reviewed
passage-level labels on the mounted project corpus, and it is not directly
leaderboard-comparable to published BEIR passage/chunk protocols.

Diagnostics now include a qrels-quality guard that checks query-to-label text
overlap. The latest NFCorpus run flags `9/30` suspicious low-overlap queries:
`deafness`, `Dr. Dean Ornish`, `Dr. Walter Willett`, `eggnog`, `energy drinks`,
`fava beans`, `Fosamax`, `How Citrus Might Help Keep Your Hands Warm`, and
`genetic manipulation`. These are not automatically invalid labels, but they
must be reviewed before treating zero-hit or low-recall behavior as purely a
retrieval/model failure.

## Remaining Stage 1 Work

The next improvement is first-stage retrieval, not reranking alone. The fixed
runs still have four zero-hit pilot queries at this candidate budget. Treat
those as qrels/content semantics risks to inspect before forcing retrieval
behavior around them.

Recommended retrieval fixes:

- Continue query expansion based on title terms and corpus-specific topic
  profiles. Use biomedical aliases only in explicit oracle diagnostics. Under
  fixed qrels, the non-oracle auto-expansion run improved candidate recall from
  `0.365927` to `0.382565`, but the repeat-spread check did not support a
  stable top-k ranking gain and did not reduce zero-hit queries.
- Keep parent-neighborhood expansion in the Stage 1 matrix for document-level
  qrels. Under fixed qrels it improved mean candidate recall to `0.416257`, but
  the repeat-spread check did not support a stable top-10 ranking gain, so it
  should be paired with a later ordering/diversification pass rather than
  treated as a complete fix.
- Compare original NFCorpus chunking and title-preserve chunking as a dual-index
  candidate source instead of choosing only one.
- Report chunk recall and parent-document recall for every benchmark.
- Sweep candidate budgets above 250 only after measuring marginal recall gain.
- Keep Cohere Rerank in the model matrix because its score separation is now
  materially better than the heuristic baseline.
- Add qrels sanity checks for low-overlap public labels so the acceptance gate
  can distinguish retrieval failures from label/chunk-transfer mismatches.

## Product Workflow Status

```text
Research Question
  -> Research Frame Constructor
  -> Semantic Retrieval Engine
  -> Evidence Neighborhood Mapper
  -> Concept & Discipline Analyzer
  -> Research Guidance Layer
  -> Human Revision
  -> Iterative Retrieval
  -> Evidence Synthesis
```

Current status by layer:

- Research Question: partially implemented through query and benchmark inputs.
- Research Frame Constructor: implemented with deterministic parsing and an
  optional OpenRouter enrichment path for less deterministic framing.
- Semantic Retrieval Engine: implemented and actively evaluated with BM25,
  dense embeddings, corpus topic profiles, fusion, reranking, and public qrels.
- Evidence Neighborhood Mapper: partially implemented through parent-context,
  source diversity, graph clusters, and parent-dominance diagnostics.
- Concept & Discipline Analyzer: partially designed in docs; needs production
  classification and evaluation slices.
- Research Guidance Layer: partially implemented through diagnostics,
  recommendations, and pre-human gates.
- Layers 1-6 can now be run together through `canon.product.research_workflow`
  or `POST /v1/research-workflow`. This deterministic production path parses
  the research question, constructs a reviewable frame, retrieves evidence
  packets, maps the evidence neighborhood, tags concepts/disciplines, and emits
  next actions before synthesis.
- Human Revision: workflow exists through qrels review and pilot review CSVs,
  but Stage 1 is currently running without new human review.
- Iterative Retrieval: optimizer, sweeps, and diagnostics now support iteration.
- Evidence Synthesis: implemented elsewhere in CANON, but should remain blocked
  from final claims until retrieval and human-reviewed labels pass.

## Stage 2 Build Status

Stage 2 now has a deterministic evidence-synthesis product path through
`canon.product.stage2_synthesis`, `canon-stage2-synthesis`, and
`POST /v1/stage2-synthesis`.

It consumes the Stage 1 research workflow, carries forward compact supporting
evidence, creates many-to-many cited synthesis claims, and runs automated
quality gates for:

- evidence presence
- claim presence
- independent source breadth
- citation integrity
- claim-to-evidence overlap
- Stage 1 workflow status visibility

The Stage 2 claim model now includes `EvidenceLink` and `SynthesizedClaim`
structures. Each claim can carry multiple evidence links, and each link records
stance (`supports`, `contradicts`, `qualifies`, `neutral`, or `undetermined`),
an entailment-style score, a hedge score, and an excerpt span.

Stage 2 now has two synthesis modes:

- deterministic local mode, used for offline tests and fallback behavior;
- model-backed mode through `stage2_model_provider` / `stage2_model`, currently
  supporting OpenRouter and OpenAI JSON synthesis.

Model-backed Stage 2 extracts atomic claims, clusters opposing evidence into
shared claim structures, assigns stance and hedge scores, and emits the same
many-to-many disagreement-map contract. External model calls are blocked unless
`allow_external_stage2_data=true` is explicitly set, because evidence text may
come from a private mounted corpus. A real OpenRouter smoke test using only
synthetic evidence produced one contested claim with both supporting and
contradicting evidence links.

Stage 2 also emits a `disagreement_map` with claim clusters, stance groups,
net support, contested-cluster counts, and disagreement axes such as
measurement/operationalization, population/scope, mechanism/theory, and
temporal scope. This turns the benchmark target from paragraph fluency into a
testable structure: whether CANON preserves contested evidence instead of
flattening it.

The current status boundary is `ready_for_human_review` when automated checks
pass, or `blocked_insufficient_evidence` / `blocked_quality_gate` when they do
not. This lets Stage 2 move forward before human review while still preventing
unsupported final claims.

The positive demo run for `ai_infra_geo_risk_demo` on grid-risk synthesis
produced four cited claims across three sources and passed the automated gate as
`pass_pending_human_review`. A broader Latin America risk query correctly
blocked because the evidence packet was empty.

## Completion Definition

Stage 1 should be considered complete when:

- the public multi-topic benchmark can run end to end;
- the mounted project corpus has at least 30 human-reviewed questions;
- qrels semantics are explicit for every corpus;
- candidate recall and work-level recall are reported together;
- learned reranker score distributions are logged;
- parent duplication is measured and controlled;
- the Stage 1 gate passes or the failure reasons are accepted as documented
  product limitations.
