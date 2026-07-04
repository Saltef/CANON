# Phase 18: Technical Probe Suite

Phase 18 adds adversarial, annotation-free probes for the named social science
RAG corpus. The probe suite is meant to expose behavior that broad averages can
hide:

- disagreement handling
- relevance traps
- cross-topic diversity pressure
- recency pressure
- out-of-domain weak-support behavior
- method-focused retrieval

Run it with:

```powershell
python -m canon.eval.probes --mode social_science_ir_v1_harvest10 --method-ids diverse_k5_template,rag_k5_template
```

The Docker target is:

```powershell
make eval-probes
```

Inputs live in `gold/probe_queries.json`. Results are written to
`reports/probe_eval_<mode>_<method_set_id>.json`.

The checks are technical expectations over existing metrics such as context
relevance, semantic alignment, claim coverage, source diversity, cluster
diversity, citation support, and conflict awareness. No human annotation is
required.

Follow-on technical runners:

```powershell
python -m canon.eval.batches --mode social_science_ir_v1_harvest10 --batch-sizes 5,10,25,50
python -m canon.eval.providers --mode social_science_ir_v1_harvest10 --providers local,openai,cohere
python -m canon.embeddings.pgvector --mode social_science_ir_v1_harvest10 --provider local
python -m canon.fulltext.grobid --mode social_science_ir_v1_harvest10
python -m canon.eval.tuning --mode social_science_ir_v1_harvest10
python -m canon.reports.dashboard --mode social_science_ir_v1_harvest10
python -m canon.experiments.manifest --mode social_science_ir_v1_harvest10 --experiment-id social_science_ir_v1_harvest10_full
```
