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
- Parent cap: `max_chunks_per_parent=1`

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

The automated gate still reports `automated_fail` because broad public qrels
make full candidate recall unrealistic at this candidate budget. That is useful
signal, not a plumbing failure: four queries have no relevant evidence in the
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

## Oracle Reranking Diagnosis

The previous heuristic reranker had weak discrimination:

- relevant score mean: `0.133563`
- non-relevant score mean: `0.093827`
- AUC-like separation: `0.556983`
- overlap rate: `0.886035`
- interpretation: `heavy_overlap_or_weak_discrimination`

Cohere Rerank with the parent cap gives much stronger observability:

- relevant score mean: `0.353832`
- non-relevant score mean: `0.254203`
- AUC-like separation: `0.95575`
- overlap rate: `0.0885`
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

## Remaining Stage 1 Work

The next improvement is first-stage retrieval, not reranking alone. The
remaining zero-hit queries are:

- `deafness`
- `energy drinks`
- `fava beans`
- `genetic manipulation`

Recommended retrieval fixes:

- Add query expansion based on biomedical aliases and title terms.
- Compare original NFCorpus chunking and title-preserve chunking as a dual-index
  candidate source instead of choosing only one.
- Report chunk recall and parent-document recall for every benchmark.
- Sweep candidate budgets above 250 only after measuring marginal recall gain.
- Keep Cohere Rerank in the model matrix because its score separation is now
  materially better than the heuristic baseline.

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
- Research Frame Constructor: designed, but needs structured frame output for
  topic, scope, methods, concepts, and inclusion/exclusion criteria.
- Semantic Retrieval Engine: implemented and actively evaluated with BM25,
  dense embeddings, fusion, reranking, and public qrels.
- Evidence Neighborhood Mapper: partially implemented through parent-context,
  source diversity, graph clusters, and parent-dominance diagnostics.
- Concept & Discipline Analyzer: partially designed in docs; needs production
  classification and evaluation slices.
- Research Guidance Layer: partially implemented through diagnostics,
  recommendations, and pre-human gates.
- Human Revision: workflow exists through qrels review and pilot review CSVs,
  but Stage 1 is currently running without new human review.
- Iterative Retrieval: optimizer, sweeps, and diagnostics now support iteration.
- Evidence Synthesis: implemented elsewhere in CANON, but should remain blocked
  from final claims until retrieval and human-reviewed labels pass.

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
