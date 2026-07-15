# Evidence-Grounded Intelligence Runner

CANON includes a deterministic mock intelligence runner that consumes evidence
packets and produces a review-ready intelligence brief. It exists to test the
integration contract between evidence retrieval and downstream analysis.

The runner is intentionally not an autonomous research system. It does not
discover new sources, rewrite citations, or claim final truth. Its job is to
prove that downstream agent-style interpretation can stay grounded in packet
evidence.

## What It Enforces

- At least five analyst roles are planned for a full brief.
- Agent claims must carry evidence IDs from the packet response.
- The red-team pass blocks factual claims with missing or unknown evidence IDs.
- Duplicate agent output is measured and surfaced.
- Source gaps, frame coverage gaps, uncertainty, and citation appendices remain
  visible in the final brief.

## Offline Demo

```powershell
python -m canon.ingest.unstructured --input data/fixtures/ai_infra_geo_risk_sample.jsonl --mode ai_infra_geo_risk_demo --chunk-tokens 80 --overlap-tokens 10
python -m canon.intelligence.evidence_runner "What are the emerging geopolitical risks around AI data center expansion in Latin America?" --mode ai_infra_geo_risk_demo --policy rag
```

The command writes JSON and Markdown reports under `reports/`.

## Product API

Start the local API:

```powershell
python -m canon.product.server --host 127.0.0.1 --port 8000
```

Then request a grounded brief:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/v1/intelligence-brief -ContentType "application/json" -Body '{"query":"What are the emerging geopolitical risks around AI data center expansion in Latin America?","mode":"ai_infra_geo_risk_demo","policy":"rag","write_report":true}'
```

The response includes `grounding_report`, `red_team`, `report_quality`, and the
final `brief`.

## Evaluation Gate

Run the intelligence-brief gate over seed questions:

```powershell
python -m canon.eval.intelligence_brief --mode ai_infra_geo_risk_demo --queries-path gold/ai_infra_geo_risk_seed_queries.json --policy rag
```

The evaluation report checks:

- grounded claim ratio >= 0.95
- at least five completed agents
- duplicate agent output rate <= 0.20
- no red-team blockers
- required report sections and citation appendix present

It writes `reports/intelligence_brief_eval_<mode>_<policy>.json` and `.md`.

## Alert Digest

Generate evidence-triggered alert prompts:

```powershell
python -m canon.intelligence.alerts "What are the emerging geopolitical risks around AI data center expansion in Latin America?" --mode ai_infra_geo_risk_demo --policy rag
```

Evaluate alert digests across seed questions:

```powershell
python -m canon.eval.alert_digest --mode ai_infra_geo_risk_demo --queries-path gold/ai_infra_geo_risk_seed_queries.json --policy rag
```

Alerts include an evidence trigger, evidence IDs or explicit source-gap status,
affected region/entity/issue, confidence, uncertainty, and recommended
follow-up. The automated gate checks structure and duplicate rate only; human
review is still required for usefulness and severity calibration.

## Human Review Packet

Prepare review tasks from the seed questions:

```powershell
python -m canon.product.intelligence_review --prepare-review --mode ai_infra_geo_risk_demo --queries-path gold/ai_infra_geo_risk_seed_queries.json --policy rag
```

Export the review fields to CSV:

```powershell
python -m canon.product.intelligence_review --export-review-csv --records reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.json
```

After a human reviewer fills the labels, import them:

```powershell
python -m canon.product.intelligence_review --import-review-csv reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.review.csv --records reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.json --output reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.completed.json
```

Then check completion:

```powershell
python -m canon.product.intelligence_review --review-status --records reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.completed.json
```

Review labels include usefulness, actionability, evidence trust, uncertainty
clarity, missing perspective, unsupported claim, overclaim risk, final status,
and reviewer notes.

## Human Review Boundary

The output status `ready_for_human_review` means the grounding checks passed.
It does not mean the report is publication-ready. Human review remains required
for usefulness, factual correctness, missing perspectives, source quality, and
overclaim risk.
