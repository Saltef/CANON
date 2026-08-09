# LlamaIndex vs CANON Factor Correction

CANON includes an internal LlamaIndex baseline for the Stage 1 NFCorpus
fixed-qrels protocol. The original head-to-head framing is superseded: it
compared CANON with Qwen embeddings, BM25, RRF, and Cohere rerank against
LlamaIndex with a hashed deterministic embedding adapter and no fusion or
rerank. That was not evidence that CANON's architecture earned its complexity.

The corrected benchmark is a 2x2 retrieval-factor pilot:

> With the same 30-query qrels protocol, how much of the gap comes from the
> embedding model, how much from the retrieval pipeline, and how much from their
> interaction?

## Protocol

- Corpus: `beir_nfcorpus_stage1_title_preserve`
- Qrels: `gold/beir_nfcorpus_stage1_title_preserve_qrels.json`
- Query count: 30
- Label count: 1,724
- Repeats: 3 per configuration
- Candidate budget: 250 for LlamaIndex single dense retrieval; 250 BM25 plus
  250 dense candidates for CANON RRF configurations
- CANON full pipeline: BM25 + dense retrieval + RRF + `cohere:rerank-v4.0-fast`
- CANON ablation: BM25 + dense retrieval + RRF without rerank
- LlamaIndex path: `llama-index-core` `VectorStoreIndex` with explicit CANON
  embedding adapters and `index.as_retriever(similarity_top_k=250)`
- Machine-readable artifact: `reports/llamaindex_stage1_head_to_head.json`

The human-review boundary still applies. This is a public-qrels retrieval pilot,
not a passage-level human-reviewed acceptance set, a generation-quality result,
or a full BEIR leaderboard claim.

## Result

| Encoder | LlamaIndex single dense nDCG@10 | CANON BM25+dense RRF+Cohere nDCG@10 |
|---|---:|---:|
| `hashed-semantic-v1` | 0.128390 | 0.350162 |
| `qwen/qwen3-embedding-8b` | 0.465538 | 0.395260 |

The original large gap was mostly an encoder confound. With the stronger Qwen
encoder in both arms, LlamaIndex's single dense retriever beats the current
CANON full pipeline by 7.028 percentage points on nDCG@10 in this pilot.

## Diagnostic Readout

The corrected report now includes compact per-query factor diagnostics. With
Qwen shared, LlamaIndex wins 13 queries, CANON wins 6, and 11 are ties. The
dominant failure classes are:

- `encoder_rescues_llamaindex`: 23 queries
- `llamaindex_qwen_ranking_advantage`: 12 queries
- `cohere_rerank_helps_qwen`: 9 queries
- `cohere_rerank_hurts_qwen`: 8 queries
- `shared_candidate_miss`: 4 queries

The useful engineering reading is that encoder strength dominates this slice,
and CANON's extra stages are not yet consistently earning their complexity once
Qwen is shared.

## Paired Deltas

| Comparison | Runs | Mean nDCG@10 delta | Spread | Standing |
|---|---:|---:|---:|---|
| LlamaIndex Qwen - LlamaIndex hash | 3 | +33.715 pp | 0.009 pp | `stands_under_repeat_spread_check` |
| CANON Qwen+Cohere - CANON hash+Cohere | 3 | +4.510 pp | 0.101 pp | `stands_under_repeat_spread_check` |
| CANON hash+Cohere - LlamaIndex hash | 3 | +22.177 pp | 0.000 pp | `stands_under_repeat_spread_check` |
| CANON Qwen+Cohere - LlamaIndex Qwen | 3 | -7.028 pp | 0.103 pp | `stands_under_repeat_spread_check` |

## Reranker Ablation

| Encoder | CANON RRF-only nDCG@10 | CANON + Cohere nDCG@10 | Rerank delta |
|---|---:|---:|---:|
| `hashed-semantic-v1` | 0.245204 | 0.350162 | +10.496 pp |
| `qwen/qwen3-embedding-8b` | 0.383108 | 0.395260 | +1.215 pp |

The reranker helps substantially with the weak hash encoder and only modestly
with Qwen on this 30-query slice. That is the useful signal: strong dense
retrieval absorbs much of what CANON's later stages were buying here.

## Boundary

Do not describe this artifact as "CANON beats LlamaIndex." The corrected
standing verdict is `corrected_factorial_result_available`: publish it as a
2x2 retrieval-factor pilot and as an example of how CANON catches confounded
claims.

Official LlamaIndex embedding defaults are version-sensitive, so this benchmark
passes explicit CANON/OpenRouter embedding adapters. It does not claim to
measure an implicit hosted provider default.

## Run It

```powershell
python -m pip install -e ".[baselines]"
python -m canon.baselines.llamaindex_baseline --repeats 3 --no-resume
```

The installed CLI entry point is:

```powershell
canon-llamaindex-baseline --repeats 3 --no-resume
```

Use `--include-query-details` only when debugging individual rankings; the
default committed JSON is compact.
