from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.alert_digest import evaluate_alert_digests
from canon.eval.intelligence_brief import DEFAULT_QUERY_PATH, evaluate_intelligence_briefs, resolve_path
from canon.ingest.unstructured import ingest_unstructured_jsonl
from canon.intelligence.alerts import run_alert_digest
from canon.intelligence.evidence_runner import run_intelligence_brief
from canon.product.intelligence_review import build_review_handoff, build_review_status_report, build_review_tasks
from canon.product.report_io import write_json, write_markdown


REPORT_ID = "flagship_handoff_v1"
DEFAULT_MODE = "ai_infra_geo_risk_demo"
DEFAULT_FIXTURE = Path("data/fixtures/ai_infra_geo_risk_sample.jsonl")
DEFAULT_QUESTION = "What are the emerging geopolitical risks around AI data center expansion in Latin America?"


def run_flagship_handoff(
    mode: str = DEFAULT_MODE,
    question: str = DEFAULT_QUESTION,
    fixture_path: Path | None = None,
    queries_path: Path | None = None,
    policy: str = "rag",
    project_id: str = "ai_infra_geo_risk",
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    fixture = resolve_path(settings.root, fixture_path or DEFAULT_FIXTURE)
    queries = resolve_path(settings.root, queries_path or DEFAULT_QUERY_PATH)
    ingest_report = ingest_unstructured_jsonl(fixture, mode=mode, chunk_tokens=80, overlap_tokens=10)
    brief_report = run_intelligence_brief(
        question=question,
        mode=mode,
        policy=policy,
        project_id=project_id,
        write_report=write_report,
    )
    brief_eval = evaluate_intelligence_briefs(
        mode=mode,
        queries_path=queries,
        policy=policy,
        project_id=project_id,
        write_report=write_report,
    )
    alert_digest = run_alert_digest(
        question=question,
        mode=mode,
        policy=policy,
        project_id=project_id,
        write_report=write_report,
    )
    alert_eval = evaluate_alert_digests(
        mode=mode,
        queries_path=queries,
        policy=policy,
        project_id=project_id,
        write_report=write_report,
    )
    review_packet = build_review_tasks(
        mode=mode,
        queries_path=queries,
        policy=policy,
        project_id=project_id,
        write_report=write_report,
    )
    review_records_path = settings.reports_dir / f"intelligence_brief_review_tasks_{mode}.json"
    review_status = (
        build_review_status_report(review_records_path, write_report=write_report)
        if write_report
        else review_status_from_packet(review_packet)
    )
    if write_report:
        build_review_handoff(review_records_path, write_report=write_report)
    steps = step_results(
        ingest_report=ingest_report,
        brief_report=brief_report,
        brief_eval=brief_eval,
        alert_digest=alert_digest,
        alert_eval=alert_eval,
        review_packet=review_packet,
        review_status=review_status,
    )
    automated_pass = all(step["passed"] for step in steps if step["id"] != "human_review_status")
    human_review_complete = next(step for step in steps if step["id"] == "human_review_status")["passed"]
    report = {
        "report_id": REPORT_ID,
        "mode": mode,
        "policy": policy,
        "project_id": project_id,
        "question": question,
        "status": handoff_status(automated_pass, human_review_complete),
        "steps": steps,
        "human_review_boundary": {
            "required": True,
            "status": review_status.get("status"),
            "reviewed_question_count": review_status.get("reviewed_question_count", 0),
            "question_count": review_status.get("question_count", 0),
            "next_required_action": next_required_action(automated_pass, review_status),
        },
        "artifacts": artifact_manifest(settings.reports_dir, mode, question, policy),
        "claim": handoff_claim(automated_pass, human_review_complete),
        "limitations": [
            "This handoff proves the offline flagship workflow and automated gates, not final report quality.",
            "Human review labels remain required before usefulness, factual-correctness, or alert-precision claims.",
            "The fixture corpus is small and synthetic-example based; connect a real mounted corpus for substantive use.",
        ],
    }
    if write_report:
        write_json(settings.reports_dir / f"flagship_handoff_{mode}.json", report)
        report["artifacts"] = artifact_manifest(settings.reports_dir, mode, question, policy)
        write_json(settings.reports_dir / f"flagship_handoff_{mode}.json", report)
        write_markdown(settings.reports_dir / f"flagship_handoff_{mode}.md", render_markdown(report))
    return report


def step_results(
    ingest_report: dict[str, Any],
    brief_report: dict[str, Any],
    brief_eval: dict[str, Any],
    alert_digest: dict[str, Any],
    alert_eval: dict[str, Any],
    review_packet: dict[str, Any],
    review_status: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        step("ingest_fixture", ingest_report.get("chunk_count", 0) > 0, ingest_report),
        step("intelligence_brief", brief_report.get("status") == "ready_for_human_review", brief_report),
        step("intelligence_brief_eval", brief_eval.get("status") == "pass", brief_eval),
        step("alert_digest", alert_digest.get("status") == "ready_for_human_review", alert_digest),
        step("alert_digest_eval", alert_eval.get("status") == "pass", alert_eval),
        step("human_review_packet", len(review_packet.get("records") or []) > 0, review_packet),
        step("human_review_status", review_status.get("status") == "complete", review_status),
    ]


def step(identifier: str, passed: bool, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": identifier,
        "status": report.get("status", "ok" if passed else "missing"),
        "passed": bool(passed),
        "summary": compact_summary(identifier, report),
    }


def compact_summary(identifier: str, report: dict[str, Any]) -> dict[str, Any]:
    if identifier == "ingest_fixture":
        return {
            "work_count": report.get("work_count", 0),
            "chunk_count": report.get("chunk_count", 0),
            "limitations": report.get("limitations", []),
        }
    if identifier == "intelligence_brief":
        return {
            "grounding_status": (report.get("grounding_report") or {}).get("status"),
            "quality_status": (report.get("report_quality") or {}).get("status"),
            "agent_count": len(report.get("agent_outputs") or []),
        }
    if identifier == "intelligence_brief_eval":
        return report.get("summary", {})
    if identifier == "alert_digest":
        return {
            "alert_count": report.get("alert_count", 0),
            "duplicate_alert_rate": (report.get("duplicate_alerts") or {}).get("duplicate_alert_rate"),
        }
    if identifier == "alert_digest_eval":
        return report.get("summary", {})
    if identifier == "human_review_packet":
        return {"record_count": len(report.get("records") or [])}
    if identifier == "human_review_status":
        return {
            "question_count": report.get("question_count", 0),
            "reviewed_question_count": report.get("reviewed_question_count", 0),
            "missing_field_count": report.get("missing_field_count", 0),
            "validation_error_count": report.get("validation_error_count", 0),
        }
    return {}


def review_status_from_packet(packet: dict[str, Any]) -> dict[str, Any]:
    records = packet.get("records") or []
    return {
        "report_id": "intelligence_brief_review_status_v1",
        "status": "incomplete",
        "question_count": len(records),
        "reviewed_question_count": 0,
        "missing_field_count": len(records) * 8,
        "validation_error_count": 0,
        "validation_errors": [],
        "rows": [],
    }


def handoff_status(automated_pass: bool, human_review_complete: bool) -> str:
    if automated_pass and human_review_complete:
        return "pass"
    if automated_pass:
        return "automated_pass_human_review_required"
    return "fail"


def handoff_claim(automated_pass: bool, human_review_complete: bool) -> str:
    if automated_pass and human_review_complete:
        return "Flagship handoff passed automated gates and human review."
    if automated_pass:
        return "Flagship handoff passed automated gates; human review remains required."
    return "Flagship handoff has failing automated gates."


def next_required_action(automated_pass: bool, review_status: dict[str, Any]) -> str:
    if not automated_pass:
        return "Fix failing automated handoff steps before human review."
    if review_status.get("status") != "complete":
        remaining = int(review_status.get("question_count") or 0) - int(review_status.get("reviewed_question_count") or 0)
        return f"Complete human review labels for {remaining} intelligence brief review task(s)."
    return "No required action remains."


def artifact_manifest(reports_dir: Path, mode: str, question: str, policy: str) -> list[dict[str, Any]]:
    paths = [
        reports_dir / f"flagship_handoff_{mode}.json",
        reports_dir / f"intelligence_brief_eval_{mode}_{policy}.json",
        reports_dir / f"alert_digest_eval_{mode}_{policy}.json",
        reports_dir / f"intelligence_brief_review_tasks_{mode}.json",
        reports_dir / f"intelligence_brief_review_tasks_{mode}.review.csv",
        reports_dir / f"intelligence_brief_review_status_v1.json",
        reports_dir / f"intelligence_brief_review_handoff_v1.json",
        reports_dir / f"intelligence_brief_review_handoff_v1.md",
    ]
    paths.extend(sorted(reports_dir.glob(f"intelligence_brief_{mode}_*.json"))[:1])
    paths.extend(sorted(reports_dir.glob(f"alert_digest_{mode}_*.json"))[:1])
    return [fingerprint(path, reports_dir) for path in paths]


def fingerprint(path: Path, reports_dir: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "id": artifact_id(path),
            "path": relative_path(path, reports_dir),
            "exists": False,
            "size_bytes": 0,
            "sha256": None,
        }
    data = path.read_bytes()
    return {
        "id": artifact_id(path),
        "path": relative_path(path, reports_dir),
        "exists": True,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def artifact_id(path: Path) -> str:
    return path.name.replace(".", "_").replace("-", "_")


def relative_path(path: Path, reports_dir: Path) -> str:
    try:
        return str(path.relative_to(reports_dir.parent)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def render_markdown(report: dict[str, Any]) -> str:
    steps = "\n".join(
        f"- {'PASS' if step['passed'] else 'BLOCKED'} `{step['id']}` observed={step['status']}"
        for step in report["steps"]
    )
    artifacts = "\n".join(
        f"- `{item['path']}` sha256={item['sha256'] or 'missing'} size={item['size_bytes']}"
        for item in report["artifacts"]
    )
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# Flagship Handoff

Status: **{report['status']}**

Question: {report['question']}

Claim: {report['claim']}

Next required action: {report['human_review_boundary']['next_required_action']}

## Steps

{steps}

## Artifacts

{artifacts}

## Limitations

{limitations}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CANON flagship handoff workflow.")
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--queries-path", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--policy", default="rag")
    parser.add_argument("--project-id", default="ai_infra_geo_risk")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    report = run_flagship_handoff(
        mode=args.mode,
        question=args.question,
        fixture_path=args.fixture,
        queries_path=args.queries_path,
        policy=args.policy,
        project_id=args.project_id,
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
