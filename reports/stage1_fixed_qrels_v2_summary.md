# Stage 1 Fixed-Qrels V2 Summary

Status: `pilot_result_not_leaderboard_comparable`

This report replaces the older candidate-dependent `parent_qrels` numbers. In
the fixed protocol, `parent_qrels=true` expands document-level BEIR labels once
from the full corpus chunk-to-parent map before candidate output is inspected.
The effective qrels are no longer a function of the retrieval configuration.

| Configuration | Qrels scope | Candidate recall mean | Work-level hit rate mean | nDCG@10 | Recall@10 | MAP@10 | MRR@10 | Zero-hit queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen+Cohere | original chunk qrels | 0.365927 | 0.395415 | 0.393658 | 0.142102 | 0.294536 | 0.647222 | 4/30 |
| Auto query expansion + fixed parent qrels | full corpus parent map v2 | 0.382565 | 0.418675 | 0.395759 | 0.142102 | 0.296783 | 0.650000 | 4/30 |
| Auto query expansion + fixed parent qrels + parent-neighborhood candidates | full corpus parent map v2 | 0.416257 | 0.418675 | 0.393091 | 0.142102 | 0.293396 | 0.650000 | 4/30 |

Interpretation:

- The corrected single-run auto-expansion gain is tiny and should not be
  claimed without the repeat-spread report.
- The 3x repeat-spread check found direction-unstable paired nDCG@10 deltas
  versus base, so the current pilot does not support a top-k ranking gain claim.
- Parent-neighborhood expansion improves candidate coverage but does not
  improve top-10 ranking in this pilot.
- This 30-query NFCorpus slice is useful for regression and product debugging,
  not for claims against published BEIR leaderboards.
- Human-reviewed passage-level qrels are still required for publication-quality
  claims.

Commands and full metadata are in
`reports/stage1_fixed_qrels_v2_summary.json`.
The repeat-spread report is in
`reports/stage1_fixed_qrels_v2_repeat_spread.json`.
