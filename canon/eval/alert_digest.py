from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.intelligence_brief import DEFAULT_QUERY_PATH, load_queries, relative, resolve_path
from canon.intelligence.alerts import run_alert_digest
from canon.product.report_io import write_json, write_markdown


REPORT_ID = "alert_digest_eval_v1"
MAX_DUPLICATE_ALERT_RATE = 0.10


def evaluate_alert_digests(
    mode: str,
    queries_path: Path | None = None,
    policy: str = "rag",
    project_id: str = "ai_infra_geo_risk",
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    resolved_queries_path = resolve_path(settings.root, queries_path or DEFAULT_QUERY_PATH)
    rows = []
    for query in load_queries(resolved_queries_path):
        digest = run_alert_digest(
            question=query["query"],
            mode=mode,
            policy=policy,
            project_id=project_id,
            write_report=False,
        )
        rows.append(evaluate_one_digest(query, digest))
    summary = summarize_rows(rows)
    report = {
        "report_id": REPORT_ID,
        "mode": mode,
        "policy": policy,
        "project_id": project_id,
        "queries_path": relative(settings.root, resolved_queries_path),
        "status": "pass" if summary["all_alert_gates_pass"] else "review",
        "thresholds": {"max_duplicate_alert_rate": MAX_DUPLICATE_ALERT_RATE},
        "summary": summary,
        "queries": rows,
        "human_review_boundary": {
            "required_before": [
                "claiming alert usefulness",
                "claiming high-severity alert precision",
                "sending operational notifications",
            ],
            "review_fields": ["alert_usefulness", "severity_calibration", "duplicate_alert", "missed_alert"],
        },
        "limitations": [
            "This evaluates alert structure and duplicate rate, not real monitoring precision.",
            "Seed questions are small; human-reviewed alert timelines are required before production alert claims.",
        ],
    }
    if write_report:
        stem = f"alert_digest_eval_{mode}_{policy}"
        write_json(settings.reports_dir / f"{stem}.json", report)
        write_markdown(settings.reports_dir / f"{stem}.md", render_markdown(report))
    return report


def evaluate_one_digest(query: dict[str, Any], digest: dict[str, Any]) -> dict[str, Any]:
    quality = digest.get("quality") or {}
    duplicate = digest.get("duplicate_alerts") or {}
    duplicate_rate = float(duplicate.get("duplicate_alert_rate") or 0.0)
    checks = [
        check("digest_ready_for_human_review", digest.get("status") == "ready_for_human_review"),
        check("alert_quality_pass", quality.get("status") == "pass"),
        check("alerts_present", int(digest.get("alert_count") or 0) > 0),
        check("duplicate_alert_rate", duplicate_rate <= MAX_DUPLICATE_ALERT_RATE),
        check("all_alerts_have_required_fields", all(alert_has_required_fields(alert) for alert in digest.get("alerts") or [])),
    ]
    return {
        "id": query.get("id"),
        "type": query.get("type"),
        "query": query.get("query"),
        "status": "pass" if all(item["passed"] for item in checks) else "review",
        "checks": checks,
        "metrics": {
            "alert_count": digest.get("alert_count", 0),
            "duplicate_alert_rate": duplicate_rate,
            "quality_status": quality.get("status"),
        },
    }


def alert_has_required_fields(alert: dict[str, Any]) -> bool:
    required = ["alert_id", "severity", "confidence", "issue", "evidence_trigger", "uncertainty", "recommended_follow_up"]
    if not all(alert.get(field) for field in required):
        return False
    return bool(alert.get("evidence_ids")) or alert.get("issue") == "source_gap"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_count = len(rows)
    pass_count = sum(1 for row in rows if row["status"] == "pass")
    alert_counts = [int(row["metrics"].get("alert_count") or 0) for row in rows]
    duplicate_rates = [float(row["metrics"].get("duplicate_alert_rate") or 0.0) for row in rows]
    return {
        "query_count": query_count,
        "pass_count": pass_count,
        "pass_rate": round(float(pass_count) / float(query_count), 4) if query_count else 0.0,
        "total_alert_count": sum(alert_counts),
        "average_alert_count": round(sum(alert_counts) / len(alert_counts), 4) if alert_counts else 0.0,
        "max_duplicate_alert_rate": round(max(duplicate_rates, default=0.0), 4),
        "all_alert_gates_pass": query_count > 0 and pass_count == query_count,
    }


def check(identifier: str, passed: bool) -> dict[str, Any]:
    return {"id": identifier, "passed": bool(passed)}


def render_markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        f"- `{row['id']}`: {row['status']} alerts={row['metrics']['alert_count']} "
        f"duplicate={row['metrics']['duplicate_alert_rate']}"
        for row in report["queries"]
    )
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# Alert Digest Evaluation

Status: **{report['status']}**

Mode: `{report['mode']}`

Queries: {report['summary']['query_count']}

Total alerts: {report['summary']['total_alert_count']}

Max duplicate alert rate: {report['summary']['max_duplicate_alert_rate']}

## Query Results

{rows}

## Limitations

{limitations}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate evidence-grounded alert digests.")
    parser.add_argument("--mode", default="ai_infra_geo_risk_demo")
    parser.add_argument("--queries-path", default=str(DEFAULT_QUERY_PATH))
    parser.add_argument("--policy", default="rag")
    parser.add_argument("--project-id", default="ai_infra_geo_risk")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    report = evaluate_alert_digests(
        mode=args.mode,
        queries_path=Path(args.queries_path),
        policy=args.policy,
        project_id=args.project_id,
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
