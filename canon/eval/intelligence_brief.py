from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.intelligence.evidence_runner import run_intelligence_brief
from canon.product.report_io import write_json, write_markdown


REPORT_ID = "intelligence_brief_eval_v1"
DEFAULT_QUERY_PATH = Path("gold/ai_infra_geo_risk_seed_queries.json")
DEFAULT_MIN_GROUNDED_RATIO = 0.95
DEFAULT_MIN_COMPLETED_AGENTS = 5
DEFAULT_MAX_DUPLICATE_AGENT_RATE = 0.2


def evaluate_intelligence_briefs(
    mode: str,
    queries_path: Path | None = None,
    policy: str = "rag",
    project_id: str = "ai_infra_geo_risk",
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    resolved_queries_path = resolve_path(settings.root, queries_path or DEFAULT_QUERY_PATH)
    queries = load_queries(resolved_queries_path)
    rows = []
    for query in queries:
        brief_report = run_intelligence_brief(
            question=query["query"],
            mode=mode,
            policy=policy,
            project_id=project_id,
            write_report=False,
        )
        rows.append(evaluate_one_query(query, brief_report))
    summary = summarize_rows(rows)
    report = {
        "report_id": REPORT_ID,
        "mode": mode,
        "policy": policy,
        "project_id": project_id,
        "queries_path": relative(settings.root, resolved_queries_path),
        "status": "pass" if summary["all_release_gates_pass"] else "review",
        "thresholds": {
            "min_grounded_claim_ratio": DEFAULT_MIN_GROUNDED_RATIO,
            "min_completed_agents": DEFAULT_MIN_COMPLETED_AGENTS,
            "max_duplicate_agent_output_rate": DEFAULT_MAX_DUPLICATE_AGENT_RATE,
        },
        "summary": summary,
        "queries": rows,
        "human_review_boundary": {
            "required_before": [
                "claiming analyst usefulness >= 4/5",
                "claiming factual correctness",
                "publishing unsupported-claim rates",
                "using reports for policy, legal, financial, or operational decisions",
            ],
            "review_fields": [
                "usefulness_1_5",
                "actionability_1_5",
                "evidence_trust_1_5",
                "uncertainty_clarity_1_5",
                "missing_perspective",
                "unsupported_claim",
                "overclaim_risk",
            ],
        },
        "claim": (
            "Automated intelligence brief gates passed; reports are ready for human review."
            if summary["all_release_gates_pass"]
            else "One or more automated intelligence brief gates failed or need review."
        ),
        "limitations": [
            "This evaluation checks structure and grounding, not truth or analyst usefulness.",
            "Seed questions are small and provisional; use human-reviewed report labels before release claims.",
            "The deterministic agent runner interprets packet evidence and does not discover external sources.",
        ],
    }
    if write_report:
        stem = f"intelligence_brief_eval_{mode}_{policy}"
        write_json(settings.reports_dir / f"{stem}.json", report)
        write_markdown(settings.reports_dir / f"{stem}.md", render_markdown(report))
    return report


def evaluate_one_query(query: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    grounding = report.get("grounding_report") or {}
    quality = report.get("report_quality") or {}
    duplicate = report.get("duplicate_agent_output") or {}
    red_team = report.get("red_team") or {}
    completed_agents = sum(1 for output in report.get("agent_outputs") or [] if output.get("status") == "complete")
    grounded_ratio = float(grounding.get("grounded_claim_ratio") or 0.0)
    duplicate_rate = float(duplicate.get("duplicate_agent_output_rate") or 0.0)
    checks = [
        check("status_ready_for_human_review", report.get("status") == "ready_for_human_review"),
        check("report_quality_pass", quality.get("status") == "pass"),
        check("grounding_pass", grounding.get("status") == "pass"),
        check("grounded_claim_ratio", grounded_ratio >= DEFAULT_MIN_GROUNDED_RATIO),
        check("minimum_agent_contribution", completed_agents >= DEFAULT_MIN_COMPLETED_AGENTS),
        check("duplicate_agent_output_rate", duplicate_rate <= DEFAULT_MAX_DUPLICATE_AGENT_RATE),
        check("no_red_team_blockers", int(red_team.get("blocking_issue_count") or 0) == 0),
        check("citation_appendix_present", bool((report.get("brief") or {}).get("citation_appendix"))),
    ]
    return {
        "id": query.get("id"),
        "type": query.get("type"),
        "query": query.get("query"),
        "human_review_status": query.get("human_review_status") or "not_reviewed",
        "status": "pass" if all(item["passed"] for item in checks) else "review",
        "checks": checks,
        "metrics": {
            "claim_count": grounding.get("claim_count", 0),
            "grounded_claim_ratio": grounded_ratio,
            "completed_agents": completed_agents,
            "duplicate_agent_output_rate": duplicate_rate,
            "red_team_blockers": red_team.get("blocking_issue_count", 0),
            "red_team_review_issues": red_team.get("review_issue_count", 0),
            "quality_status": quality.get("status"),
        },
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_count = len(rows)
    pass_count = sum(1 for row in rows if row["status"] == "pass")
    metrics = [row["metrics"] for row in rows]
    return {
        "query_count": query_count,
        "pass_count": pass_count,
        "pass_rate": round(safe_ratio(pass_count, query_count), 4),
        "average_grounded_claim_ratio": round(average(item["grounded_claim_ratio"] for item in metrics), 4),
        "average_completed_agents": round(average(item["completed_agents"] for item in metrics), 4),
        "max_duplicate_agent_output_rate": round(max((item["duplicate_agent_output_rate"] for item in metrics), default=0.0), 4),
        "red_team_blocker_count": sum(int(item["red_team_blockers"] or 0) for item in metrics),
        "red_team_review_issue_count": sum(int(item["red_team_review_issues"] or 0) for item in metrics),
        "all_release_gates_pass": query_count > 0 and pass_count == query_count,
    }


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of query objects.")
    queries = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict) or not row.get("query"):
            raise ValueError(f"{path}:{index}: query row must be an object with a query field.")
        queries.append(row)
    return queries


def check(identifier: str, passed: bool) -> dict[str, Any]:
    return {"id": identifier, "passed": bool(passed)}


def average(values: Any) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def render_markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        f"- `{row['id']}`: {row['status']} "
        f"(grounded={row['metrics']['grounded_claim_ratio']}, agents={row['metrics']['completed_agents']}, "
        f"duplicate={row['metrics']['duplicate_agent_output_rate']})"
        for row in report["queries"]
    )
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# Intelligence Brief Evaluation

Status: **{report['status']}**

Mode: `{report['mode']}`

Policy: `{report['policy']}`

Queries: {report['summary']['query_count']}

Pass rate: {report['summary']['pass_rate']}

Average grounded claim ratio: {report['summary']['average_grounded_claim_ratio']}

Average completed agents: {report['summary']['average_completed_agents']}

Red-team blockers: {report['summary']['red_team_blocker_count']}

## Query Results

{rows}

## Human Review Boundary

Automated checks do not prove usefulness, factual correctness, or completeness.

## Limitations

{limitations}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grounded intelligence briefs across seed questions.")
    parser.add_argument("--mode", default="ai_infra_geo_risk_demo")
    parser.add_argument("--queries-path", default=str(DEFAULT_QUERY_PATH))
    parser.add_argument("--policy", default="rag")
    parser.add_argument("--project-id", default="ai_infra_geo_risk")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    report = evaluate_intelligence_briefs(
        mode=args.mode,
        queries_path=Path(args.queries_path),
        policy=args.policy,
        project_id=args.project_id,
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
