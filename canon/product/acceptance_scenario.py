from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.intelligence.evidence_runner import DEFAULT_AI_INFRA_FRAME, DEFAULT_EVIDENCE_REQUIREMENTS
from canon.product import report_io, service
from canon.product.flagship_handoff import DEFAULT_MODE, DEFAULT_QUESTION
from canon.product.project import DEFAULT_REPORT_TYPES, build_project_config


REPORT_ID = "flagship_acceptance_scenario_v1"


def run_acceptance_scenario(
    mode: str = DEFAULT_MODE,
    question: str = DEFAULT_QUESTION,
    policy: str = "rag",
    project_id: str = "ai_infra_geo_risk",
    source_boundaries: list[str] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    source_boundaries = source_boundaries or ["data/fixtures/ai_infra_geo_risk_sample.jsonl"]
    project_config = build_project_config(
        project_name="AI Infrastructure Geopolitical Risk",
        project_id=project_id,
        domain="AI infrastructure and geopolitical risk",
        regions=DEFAULT_AI_INFRA_FRAME["regions"],
        languages=DEFAULT_AI_INFRA_FRAME["languages"],
        issue_categories=DEFAULT_AI_INFRA_FRAME["subdomains"],
        desired_report_types=DEFAULT_REPORT_TYPES,
        review_cadence="weekly",
        source_boundaries=source_boundaries,
        corpus_id=mode,
        write_report=write_report,
    )
    frame_coverage = service.frame_coverage_report(
        {
            "request_id": "acceptance_frame",
            "project_id": project_id,
            "question": question,
            "mode": mode,
            "policy": policy,
            "research_frame": DEFAULT_AI_INFRA_FRAME,
            "evidence_requirements": DEFAULT_EVIDENCE_REQUIREMENTS,
            "write_report": write_report,
        }
    )
    brief = service.intelligence_brief(
        {
            "project_id": project_id,
            "question": question,
            "mode": mode,
            "policy": policy,
            "research_frame": DEFAULT_AI_INFRA_FRAME,
            "evidence_requirements": DEFAULT_EVIDENCE_REQUIREMENTS,
            "write_report": write_report,
        }
    )
    quality_gate = service.report_quality_gate(
        {
            "project_id": project_id,
            "question": question,
            "mode": mode,
            "policy": policy,
            "write_report": write_report,
        }
    )
    alert_digest = service.alert_digest(
        {
            "project_id": project_id,
            "question": question,
            "mode": mode,
            "policy": policy,
            "write_report": write_report,
        }
    )
    review_packet = service.intelligence_review_prepare(
        {
            "project_id": project_id,
            "mode": mode,
            "policy": policy,
            "write_report": write_report,
        }
    )
    checks = acceptance_checks(project_config, frame_coverage, brief, quality_gate, alert_digest, review_packet)
    report = {
        "report_id": REPORT_ID,
        "mode": mode,
        "policy": policy,
        "project_id": project_id,
        "question": question,
        "status": "automated_pass_human_review_required" if all(row["passed"] for row in checks) else "review",
        "checks": checks,
        "source_reports": {
            "project_config_status": project_config.get("status"),
            "frame_coverage_status": frame_coverage.get("status"),
            "brief_status": brief.get("status"),
            "quality_gate_status": quality_gate.get("status"),
            "alert_digest_status": alert_digest.get("status"),
            "review_record_count": len(review_packet.get("records") or []),
        },
        "human_review_boundary": {
            "required": True,
            "reason": "Automated acceptance checks do not prove analyst usefulness, factual correctness, completeness, or alert precision.",
            "next_required_action": "Complete human review labels for reports and alerts before final acceptance claims.",
        },
        "limitations": [
            "This scenario verifies the flagship workflow shape over the current corpus.",
            "Human-reviewed qrels and reviewed reports remain required for model-quality claims.",
            "Public-opinion outputs remain caveated and must not be treated as representative polling.",
        ],
    }
    if write_report:
        report_io.write_json(settings.reports_dir / f"flagship_acceptance_scenario_{mode}.json", report)
        report_io.write_markdown(settings.reports_dir / f"flagship_acceptance_scenario_{mode}.md", render_markdown(report))
    return report


def acceptance_checks(
    project_config: dict[str, Any],
    frame_coverage: dict[str, Any],
    brief: dict[str, Any],
    quality_gate: dict[str, Any],
    alert_digest: dict[str, Any],
    review_packet: dict[str, Any],
) -> list[dict[str, Any]]:
    brief_body = brief.get("brief") or {}
    issue_categories = issue_categories_from_brief(brief)
    covered_regions = (frame_coverage.get("frame_coverage") or {}).get("coverage", {}).get("regions", {}).get("covered", [])
    return [
        check("project_boundary_ready", project_config.get("status") == "ready", {"status": project_config.get("status")}),
        check(
            "grounded_claims_and_citations",
            quality_gate.get("status") == "pass_human_review_required"
            and int((quality_gate.get("grounding") or {}).get("unsupported_claim_count") or 0) == 0,
            quality_gate.get("grounding", {}),
        ),
        check("at_least_three_issue_categories", len(issue_categories) >= 3, {"issue_categories": sorted(issue_categories)}),
        check("at_least_two_regions_or_gap_visible", len(covered_regions) >= 2 or has_gap(frame_coverage, "regions"), {"covered_regions": covered_regions}),
        check("public_opinion_or_gap_visible", public_opinion_visible(frame_coverage, brief), {}),
        check("uncertainty_section_present", bool(brief_body.get("contradictions_and_uncertainty")), {}),
        check("next_watch_signals_present", bool(brief_body.get("next_signals_to_watch")), {}),
        check("alert_digest_ready", alert_digest.get("status") == "ready_for_human_review", {"status": alert_digest.get("status")}),
        check("human_review_packet_created", len(review_packet.get("records") or []) > 0, {"record_count": len(review_packet.get("records") or [])}),
    ]


def issue_categories_from_brief(brief: dict[str, Any]) -> set[str]:
    categories = set()
    for output in brief.get("agent_outputs") or []:
        for claim in output.get("claims") or []:
            categories.update(str(item) for item in claim.get("issue_categories") or [] if item)
    return categories


def has_gap(report: dict[str, Any], text: str) -> bool:
    needle = text.lower()
    return any(needle in str(gap.get("gap", "")).lower() for gap in report.get("coverage_gaps") or [])


def public_opinion_visible(frame_coverage: dict[str, Any], brief: dict[str, Any]) -> bool:
    if "public opinion" in {item.lower() for item in issue_categories_from_brief(brief)}:
        return True
    coverage = (frame_coverage.get("frame_coverage") or {}).get("coverage", {})
    if "public opinion" in [item.lower() for item in coverage.get("subdomains", {}).get("covered", [])]:
        return True
    return has_gap(frame_coverage, "public opinion")


def check(identifier: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"id": identifier, "passed": bool(passed), "details": details}


def render_markdown(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- {'PASS' if row['passed'] else 'REVIEW'} `{row['id']}`"
        for row in report["checks"]
    )
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# Flagship Acceptance Scenario

Status: **{report['status']}**

Question: {report['question']}

## Checks

{checks}

## Human Review Boundary

{report['human_review_boundary']['reason']}

Next required action: {report['human_review_boundary']['next_required_action']}

## Limitations

{limitations}
"""


def csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CANON flagship acceptance scenario checklist.")
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--policy", default="rag")
    parser.add_argument("--project-id", default="ai_infra_geo_risk")
    parser.add_argument("--source-boundaries", default="")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    report = run_acceptance_scenario(
        mode=args.mode,
        question=args.question,
        policy=args.policy,
        project_id=args.project_id,
        source_boundaries=csv_list(args.source_boundaries),
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
