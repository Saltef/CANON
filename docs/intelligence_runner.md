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

Run the full fixture handoff and print a compact summary:

```powershell
python -m canon.product.demo
```

The installed console script is `canon-demo`.

For lower-level debugging, run ingestion and the brief runner separately:

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

Run a focused quality gate for one brief:

```powershell
python -m canon.product.report_quality "What are the emerging geopolitical risks around AI data center expansion in Latin America?" --mode ai_infra_geo_risk_demo --policy rag
Invoke-RestMethod -Method Post http://localhost:8000/v1/report-quality -ContentType "application/json" -Body '{"query":"What are the emerging geopolitical risks around AI data center expansion in Latin America?","mode":"ai_infra_geo_risk_demo","policy":"rag","write_report":true}'
```

This gate returns the grounding ratio, unsupported-claim count, red-team
blockers, duplicate-agent rate, required-section checks, and the human-review
boundary without requiring a reviewer to inspect the whole brief JSON first.

Run the automated intelligence-brief evaluation gate through the API:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/v1/intelligence-brief/evaluate -ContentType "application/json" -Body '{"mode":"ai_infra_geo_risk_demo","queries_path":"gold/ai_infra_geo_risk_seed_queries.json","policy":"rag","write_report":true}'
```

Generate an alert digest through the same product API:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/v1/alert-digest -ContentType "application/json" -Body '{"query":"What are the emerging geopolitical risks around AI data center expansion in Latin America?","mode":"ai_infra_geo_risk_demo","policy":"rag","write_report":true}'
```

Run the automated alert-digest evaluation gate through the API:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/v1/alert-digest/evaluate -ContentType "application/json" -Body '{"mode":"ai_infra_geo_risk_demo","queries_path":"gold/ai_infra_geo_risk_seed_queries.json","policy":"rag","write_report":true}'
```

Run the full flagship demo handoff through the API:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/v1/flagship-handoff -ContentType "application/json" -Body '{"mode":"ai_infra_geo_risk_demo","write_report":true}'
```

The flagship handoff returns `automated_pass_human_review_required` when the
offline workflow passes automated gates and still needs human labels.

Run the flagship acceptance checklist:

```powershell
python -m canon.product.acceptance_scenario --mode ai_infra_geo_risk_demo
Invoke-RestMethod -Method Post http://localhost:8000/v1/acceptance-scenario -ContentType "application/json" -Body '{"mode":"ai_infra_geo_risk_demo","write_report":true}'
```

The checklist verifies the project boundary, grounded claims/citations, at
least three issue categories, regional coverage or visible gaps, public opinion
or a public-evidence gap, uncertainty, next-watch signals, alert readiness, and
human-review packet creation. It still stops at human review.

Prepare and manage the human review packet through the API:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/v1/intelligence-review/prepare -ContentType "application/json" -Body '{"mode":"ai_infra_geo_risk_demo","queries_path":"gold/ai_infra_geo_risk_seed_queries.json","policy":"rag","write_report":true}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/intelligence-review/export-csv -ContentType "application/json" -Body '{"records_path":"reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.json","output_path":"reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.review.csv"}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/intelligence-review/import-csv -ContentType "application/json" -Body '{"records_path":"reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.json","csv_path":"reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.review.csv","output_path":"reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.completed.json"}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/intelligence-review/status -ContentType "application/json" -Body '{"records_path":"reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.completed.json"}'
Invoke-RestMethod -Method Post http://localhost:8000/v1/intelligence-review/feedback -ContentType "application/json" -Body '{"records_path":"reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.completed.json"}'
```

The import endpoint validates human-entered labels. It does not create labels or
upgrade automated results into final acceptance on its own.

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

## Flagship Handoff

Run the offline flagship workflow as one handoff report:

```powershell
python -m canon.product.flagship_handoff --mode ai_infra_geo_risk_demo
```

The handoff runner ingests the fixture corpus, generates a grounded intelligence
brief, evaluates brief quality, generates an alert digest, evaluates alert
structure, prepares human review tasks, and reports whether the workflow is
blocked only by human review.

Expected pre-human status for the fixture is:

```text
automated_pass_human_review_required
```

That status is intentional. It means the automated workflow is runnable and
review-ready, while final acceptance still belongs to the human reviewer.

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

Summarize completed labels and identify regression candidates:

```powershell
python -m canon.product.intelligence_review --feedback-report --records reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.completed.json
```

Review labels include usefulness, actionability, evidence trust, uncertainty
clarity, missing perspective, unsupported claim, overclaim risk, final status,
and reviewer notes.

## Human Review Boundary

The output status `ready_for_human_review` means the grounding checks passed.
It does not mean the report is publication-ready. Human review remains required
for usefulness, factual correctness, missing perspectives, source quality, and
overclaim risk.
