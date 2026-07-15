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

## Human Review Boundary

The output status `ready_for_human_review` means the grounding checks passed.
It does not mean the report is publication-ready. Human review remains required
for usefulness, factual correctness, missing perspectives, source quality, and
overclaim risk.
