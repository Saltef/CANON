# CANON Stage 1 Project Status

Stage 1 is the evidence-discovery and retrieval-evaluation layer. Its job is to
prove that CANON can mount a corpus, retrieve plausible evidence, rank it,
explain failures, and produce a repeatable acceptance report before human review
and synthesis claims are trusted.

## Current Stage 1 Result

The strongest focused pilot so far uses:

- Corpus: `beir_nfcorpus_stage1_title_preserve`
- Acceptance set: 30 NFCorpus public BEIR queries
- Qrels: document-level BEIR labels transferred to chunks
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

The strongest raw retrieval/ranking score now uses the same Qwen+Cohere stack
with `auto_query_expansion=true`, corpus topic-profile expansion, and
`parent_qrels=true`. The strongest coverage run also uses
`parent_expansion_limit=2`, which adds up to two sibling chunks from each
retrieved parent work into the candidate pool. This is the right diagnostic
setting for public document-level qrels and alternate chunking, because
relevance can transfer across chunks that share the same parent work.

Benchmark integrity note: hand-written biomedical alias expansion is now
quarantined behind `--benchmark-oracle-expansion`. Normal runs with
`--auto-query-expansion` use explicit query variants and corpus-derived topic
profiles only. Any run using `--benchmark-oracle-expansion` is diagnostic and
must be excluded from headline metrics.

This run produces:

- candidate recall mean: `0.365927`
- work-level hit rate mean: `0.395415`
- nDCG@10: `0.343299`
- Recall@10: `0.112004`
- MAP@10: `0.23596`
- MRR@10: `0.651111`
- queries with any relevant work in pool: `26/30`
- queries with top-10 relevant hits: `23/30`
- zero candidate-recall queries: `4/30`

With auto query expansion and parent-qrels enabled:

- candidate recall mean: `0.422138`
- work-level hit rate mean: `0.470343`
- nDCG@10: `0.397614`
- Recall@10: `0.142432`
- MAP@10: `0.298061`
- MRR@10: `0.65`
- queries with any relevant work in pool: `29/30`
- zero candidate-recall queries: `1/30`

With parent-neighborhood expansion enabled:

- candidate recall mean: `0.469410`
- work-level hit rate mean: `0.470343`
- nDCG@10: `0.396759`
- Recall@10: `0.142432`
- MAP@10: `0.297184`
- MRR@10: `0.65`
- queries with any relevant work in pool: `29/30`
- zero candidate-recall queries: `1/30`

The automated gate still reports `automated_fail` because broad public qrels
make full candidate recall unrealistic at this candidate budget. That is useful
signal, not a plumbing failure: one query still has no relevant evidence in the
candidate pool, and no reranker can recover evidence that was never retrieved.

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
sibling chunks from that work as candidate evidence. In the NFCorpus pilot this
raised candidate recall from `0.422138` to `0.469410`, which is a much larger
coverage gain than the topic-profile expansion alone.

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

Cohere Rerank with the parent cap gives much stronger observability:

- relevant score mean: `0.346928`
- non-relevant score mean: `0.245519`
- AUC-like separation: `0.856008`
- overlap rate: `0.287985`
- interpretation: `strong_discrimination`

This supports the hypothesis that the value of a learned reranker is not only
top-k movement. Its score distribution is now a real observability instrument:
low score gaps, score overlap, and low-confidence top-k results can drive review
and retrieval tuning.

## Qrels Semantics

For the NFCorpus pilot, qrels are document-level labels transferred to chunks.
The evaluation therefore reports both:

- chunk-level recall: whether relevant chunks were retrieved
- work-level recall: whether any chunk from a relevant parent document was
  retrieved

The rule is explicit: a retrieved chunk is relevant whenever its parent BEIR
document is relevant. This is appropriate for public BEIR comparison, but it is
not a substitute for human-reviewed passage-level labels on the mounted project
corpus.

Diagnostics now include a qrels-quality guard that checks query-to-label text
overlap. The latest NFCorpus run flags `9/30` suspicious low-overlap queries:
`deafness`, `Dr. Dean Ornish`, `Dr. Walter Willett`, `eggnog`, `energy drinks`,
`fava beans`, `Fosamax`, `How Citrus Might Help Keep Your Hands Warm`, and
`genetic manipulation`. These are not automatically invalid labels, but they
must be reviewed before treating zero-hit or low-recall behavior as purely a
retrieval/model failure.

## Remaining Stage 1 Work

The next improvement is first-stage retrieval, not reranking alone. The
remaining zero-hit query is `fava beans`. Its transferred gold chunk is
`chunk:81588b7b16810a61` from parent `MED-4281`, titled "Beneficial effects of
L-arginine on reducing obesity: potential mechanisms and important implications
for human health." The visible chunk text is about L-arginine, obesity, and
metabolic disorders rather than fava beans. Treat this as a qrels/content
semantics risk to inspect before forcing retrieval behavior around it.

Recommended retrieval fixes:

- Continue query expansion based on title terms and corpus-specific topic
  profiles. Use biomedical aliases only in explicit oracle diagnostics. The
  previous deterministic diagnostic expansion reduced zero-hit queries from
  `4/30` to `1/30` and improved mean candidate recall from `0.406249` to
  `0.422138`, but that gain should be remeasured under non-oracle expansion.
- Keep parent-neighborhood expansion in the Stage 1 matrix for document-level
  qrels. It improved mean candidate recall to `0.469410`, but it did not improve
  top-10 ranking, so it should be paired with a later ordering/diversification
  pass rather than treated as a complete fix.
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
