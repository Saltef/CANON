# LlamaIndex vs CANON Stage 1 Factor Correction

Status: `factorial_correction_complete`

Protocol: 30 NFCorpus queries, 1724 fixed labels, 3 repeats per cell.

Corrected conclusion: `corrected_factorial_result_available`.

## Correction

- The old -26.6pp LlamaIndex comparison should not be used as evidence that CANON's architecture earned its complexity, because encoder and pipeline changed together.
- The old primary framing compared CANON with Qwen embeddings, BM25, RRF, and Cohere rerank against LlamaIndex with a hashed embedding adapter and no fusion/rerank. That was not a pipeline-only comparison.
- Publish the result as a corrected 2x2 retrieval-factor pilot, not as proof that one framework or architecture generally wins.

## Primary 2x2

| Encoder | LlamaIndex single dense nDCG@10 | CANON BM25+dense RRF+Cohere nDCG@10 |
|---|---:|---:|
| hashed-semantic-v1 | 0.128390 | 0.350162 |
| qwen/qwen3-embedding-8b | 0.465538 | 0.395260 |

## Retrieval Diagnostics

The correction points to encoder strength as the dominant factor on this slice; CANON's multi-stage path is not yet earning its extra complexity once Qwen is shared.

### Winner Counts With Shared Qwen

{
  "canon_qwen_cohere": 6,
  "llamaindex_qwen": 13,
  "tie": 11
}

### Failure Classes

{
  "canon_candidate_recall_advantage": 1,
  "canon_multistage_advantage": 4,
  "cohere_rerank_helps_qwen": 9,
  "cohere_rerank_hurts_qwen": 8,
  "encoder_rescues_canon": 6,
  "encoder_rescues_llamaindex": 23,
  "llamaindex_qwen_ranking_advantage": 12,
  "near_tie_or_low_signal": 2,
  "shared_candidate_miss": 4
}

### Largest CANON Losses With Shared Qwen

| Query | Delta pp | Rerank pp | Classes |
|---|---:|---:|---|
| `PLAIN-1098` eggnog | -49.352 | -49.352 | `llamaindex_qwen_ranking_advantage`, `cohere_rerank_hurts_qwen`, `encoder_rescues_llamaindex` |
| `PLAIN-1151` factory farming practices | -31.597 | 49.115 | `llamaindex_qwen_ranking_advantage`, `cohere_rerank_helps_qwen`, `encoder_rescues_llamaindex`, `encoder_rescues_canon` |
| `PLAIN-1028` dietary scoring | -30.103 | 0.000 | `llamaindex_qwen_ranking_advantage`, `encoder_rescues_llamaindex` |
| `PLAIN-102` Stopping Heart Disease in Childhood | -24.812 | -5.070 | `llamaindex_qwen_ranking_advantage`, `cohere_rerank_hurts_qwen`, `encoder_rescues_llamaindex` |
| `PLAIN-1214` Fosamax | -24.736 | -24.736 | `llamaindex_qwen_ranking_advantage`, `cohere_rerank_hurts_qwen`, `encoder_rescues_llamaindex`, `encoder_rescues_canon` |

### Largest CANON Wins With Shared Qwen

| Query | Delta pp | Rerank pp | Classes |
|---|---:|---:|---|
| `PLAIN-1288` grapes | 40.318 | -6.646 | `canon_multistage_advantage`, `cohere_rerank_hurts_qwen`, `encoder_rescues_llamaindex` |
| `PLAIN-1183` Finland | 8.023 | 1.571 | `canon_multistage_advantage`, `cohere_rerank_helps_qwen`, `encoder_rescues_llamaindex` |
| `PLAIN-1193` flax oil | 6.046 | 13.886 | `canon_multistage_advantage`, `cohere_rerank_helps_qwen` |
| `PLAIN-123` How Citrus Might Help Keep Your Hands Warm | 5.598 | 8.043 | `canon_multistage_advantage`, `cohere_rerank_helps_qwen`, `encoder_rescues_llamaindex` |
| `PLAIN-112` Food Dyes and ADHD | 3.121 | 32.787 | `cohere_rerank_helps_qwen`, `encoder_rescues_llamaindex`, `encoder_rescues_canon` |

## Reranker Ablation

| Encoder | CANON RRF-only nDCG@10 | CANON RRF+Cohere nDCG@10 |
|---|---:|---:|
| hashed-semantic-v1 | 0.245204 | 0.350162 |
| qwen/qwen3-embedding-8b | 0.383108 | 0.395260 |

## Paired Deltas

| Comparison | Paired runs | Mean delta pp | Min delta pp | Max delta pp | Delta spread pp | Standing |
|---|---:|---:|---:|---:|---:|---|
| llamaindex_qwen_original_qrels - llamaindex_hash_original_qrels | 3 | 33.715 | 33.710 | 33.719 | 0.009 | `stands_under_repeat_spread_check` |
| canon_qwen_cohere_original_qrels - canon_hash_cohere_original_qrels | 3 | 4.510 | 4.444 | 4.544 | 0.101 | `stands_under_repeat_spread_check` |
| canon_hash_cohere_original_qrels - llamaindex_hash_original_qrels | 3 | 22.177 | 22.177 | 22.177 | 0.000 | `stands_under_repeat_spread_check` |
| canon_qwen_cohere_original_qrels - llamaindex_qwen_original_qrels | 3 | -7.028 | -7.094 | -6.991 | 0.103 | `stands_under_repeat_spread_check` |
| canon_hash_cohere_original_qrels - canon_hash_rrf_only_original_qrels | 3 | 10.496 | 10.496 | 10.496 | 0.000 | `stands_under_repeat_spread_check` |
| canon_qwen_cohere_original_qrels - canon_qwen_rrf_only_original_qrels | 3 | 1.215 | 1.213 | 1.218 | 0.005 | `stands_under_repeat_spread_check` |

## Stack Boundaries

- LlamaIndex: This measures LlamaIndex VectorStoreIndex retrieval with explicit embeddings, not LlamaIndex generation and not an implicit hosted provider default.
- CANON: The CANON full-pipeline cells compare BM25+dense RRF plus Cohere rerank against LlamaIndex single dense retrieval. RRF-only cells isolate reranker contribution.
- LlamaIndex default note: Current official LlamaIndex embedding defaults are version-sensitive. This benchmark therefore passes explicit CANON/OpenRouter embedding adapters and does not claim to measure an implicit hosted provider default.

The compact machine-readable report is in `reports/llamaindex_stage1_head_to_head.json`.
