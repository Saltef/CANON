# Stage 1 Fixed-Qrels V2 Repeat Spread

Status: `repeat_spread_complete`

This report repeats each fixed-qrels Stage 1 configuration 3 time(s) and reports run-to-run spread.

| Configuration | Runs | nDCG@10 mean | nDCG@10 min | nDCG@10 max | nDCG spread pp | MAP@10 mean | Candidate recall mean | Work-hit mean | Zero-hit queries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Base Qwen+Cohere | 3 | 0.394511 | 0.393720 | 0.395128 | 0.141 | 0.295470 | 0.365927 | 0.395415 | 4/30, 4/30, 4/30 |
| Auto query expansion + fixed parent qrels | 3 | 0.394649 | 0.394554 | 0.394775 | 0.022 | 0.295595 | 0.382565 | 0.418675 | 4/30, 4/30, 4/30 |
| Auto query expansion + fixed parent qrels + parent-neighborhood candidates | 3 | 0.394767 | 0.393671 | 0.396490 | 0.282 | 0.295955 | 0.416257 | 0.418675 | 4/30, 4/30, 4/30 |

## Paired nDCG@10 Deltas

| Comparison | Paired runs | Mean delta pp | Min delta pp | Max delta pp | Delta spread pp | Standing |
|---|---:|---:|---:|---:|---:|---|
| auto_query_expansion_fixed_parent_qrels - base_qwen_cohere | 3 | 0.014 | -0.057 | 0.105 | 0.163 | `does_not_stand_direction_unstable` |
| auto_query_expansion_fixed_parent_qrels_parent_neighborhood - base_qwen_cohere | 3 | 0.026 | -0.101 | 0.136 | 0.238 | `does_not_stand_direction_unstable` |

Interpretation:

- The fixed-qrels auto-expansion gain should not be claimed from this repeat check.
- Treat this as a pilot reproducibility check, not a public BEIR leaderboard result.
- Human-reviewed passage-level labels are still required for publication-quality claims.

Full per-run values and metadata are in
`reports/stage1_fixed_qrels_v2_repeat_spread.json`.
