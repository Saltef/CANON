# LlamaIndex vs CANON Head-to-Head

CANON now includes an internal LlamaIndex baseline for the Stage 1 NFCorpus
fixed-qrels protocol. The goal is not to claim framework superiority in the
abstract; it is to answer a narrower, reproducible question:

> Does an off-the-shelf LlamaIndex vector retrieval path beat CANON's current
> hand-built Stage 1 retrieval stack on the same 30-query NFCorpus pilot?

## Protocol

- Corpus: `beir_nfcorpus_stage1_title_preserve`
- Qrels: `gold/beir_nfcorpus_stage1_title_preserve_qrels.json`
- Query count: 30
- Label count: 1,724
- Repeats: 3 per LlamaIndex configuration
- CANON comparison source: `reports/stage1_fixed_qrels_v2_repeat_spread.json`
- LlamaIndex report: `reports/llamaindex_stage1_head_to_head.json`
- Human-review boundary: this remains a public-qrels retrieval pilot, not a
  passage-level human-reviewed acceptance set or a full BEIR leaderboard claim.

The LlamaIndex baseline uses `llama-index-core`'s `VectorStoreIndex` and
`index.as_retriever(similarity_top_k=candidate_k)`. CANON supplies a local
deterministic `hashed-semantic-v1` embedding adapter so the baseline can run
without sending public benchmark text to an unconfigured OpenAI default. This
means the comparison measures LlamaIndex's vector-index and retriever defaults,
not OpenAI's hosted embedding model.

## Result

| Comparison | Paired runs | Mean nDCG@10 delta | Standing |
|---|---:|---:|---|
| LlamaIndex original qrels - CANON base Qwen+Cohere | 3 | -26.627 pp | `stands_under_repeat_spread_check` |
| LlamaIndex fixed parent qrels - CANON auto expansion | 3 | -26.640 pp | `stands_under_repeat_spread_check` |
| LlamaIndex fixed parent qrels - CANON parent-neighborhood | 3 | -26.652 pp | `stands_under_repeat_spread_check` |

Standing verdict: `canon`.

This is the publishable shape of the result even though it favors CANON: a
framework baseline was added inside the same repo, scored under the same fixed
qrels and paired-repeat logic, and given the same standing-verdict treatment.

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
