# LlamaIndex vs CANON Stage 1 Head-to-Head

Status: `head_to_head_complete`

Protocol: 30 NFCorpus queries, 1724 fixed labels, 3 repeats per LlamaIndex configuration.

Standing verdict: `canon` (stands_under_repeat_spread_check, -26.627 pp on nDCG@10).

## LlamaIndex Stack

- Index: `VectorStoreIndex`
- Retriever: `index.as_retriever(similarity_top_k=candidate_k)`
- Embedding: `CANON local hashed-semantic-v1 adapter`
- Boundary: This measures LlamaIndex's default in-memory vector index and retriever plumbing. It does not use LlamaIndex's hosted OpenAI default embedding because no OpenAI key is configured and CANON keeps public benchmark reruns reproducible by default.

## LlamaIndex Repeat Spread

| Configuration | Runs | nDCG@10 mean | nDCG spread pp | MAP@10 mean | Candidate recall mean | Work-hit mean | Zero-hit queries |
|---|---:|---:|---:|---:|---:|---:|---|
| LlamaIndex VectorStoreIndex + CANON hash embeddings, original qrels | 3 | 0.128245 | 0.000 | 0.080939 | 0.199087 | 0.208370 | 7/30, 7/30, 7/30 |
| LlamaIndex VectorStoreIndex + CANON hash embeddings, fixed parent qrels | 3 | 0.128245 | 0.000 | 0.080939 | 0.199087 | 0.208370 | 7/30, 7/30, 7/30 |

## Paired nDCG@10 Deltas

| Comparison | Paired runs | Mean delta pp | Min delta pp | Max delta pp | Delta spread pp | Standing |
|---|---:|---:|---:|---:|---:|---|
| llamaindex_vector_original_qrels - base_qwen_cohere | 3 | -26.627 | -26.688 | -26.548 | 0.141 | `stands_under_repeat_spread_check` |
| llamaindex_vector_fixed_parent_qrels - auto_query_expansion_fixed_parent_qrels | 3 | -26.640 | -26.653 | -26.631 | 0.022 | `stands_under_repeat_spread_check` |
| llamaindex_vector_fixed_parent_qrels - auto_query_expansion_fixed_parent_qrels_parent_neighborhood | 3 | -26.652 | -26.825 | -26.543 | 0.282 | `stands_under_repeat_spread_check` |

## Interpretation Boundary

- The paired effect is larger than the observed within-configuration spread.
- Publish the result either way, but describe it as a 30-query NFCorpus pilot rather than a full BEIR leaderboard claim.
- This is a framework retrieval comparison, not a generation or answer-faithfulness comparison.

The compact machine-readable report is in `reports/llamaindex_stage1_head_to_head.json`.
