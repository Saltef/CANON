from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.intelligence_brief import DEFAULT_QUERY_PATH, resolve_path
from canon.intelligence.evidence_runner import run_intelligence_brief
from canon.product.report_io import write_json, write_markdown


REVIEW_PACKET_ID = "intelligence_brief_review_tasks_v1"
REVIEW_STATUS_ID = "intelligence_brief_review_status_v1"
REVIEW_CSV_FIELDS = [
    "id",
    "type",
    "query",
    "brief_status",
    "executive_summary",
    "what_changed",
    "source_gaps",
    "red_team_objections",
    "citation_appendix",
    "evidence_scope_summary",
    "packet_contract_status",
    "external_expansion_status",
    "external_expansion_executed",
    "grounded_claim_ratio",
    "duplicate_agent_output_rate",
    "completed_agents",
    "usefulness_1_5",
    "actionability_1_5",
    "evidence_trust_1_5",
    "uncertainty_clarity_1_5",
    "missing_perspective",
    "unsupported_claim",
    "overclaim_risk",
    "final_review_status",
    "reviewer_notes",
]

REQUIRED_REVIEW_FIELDS = [
    "usefulness_1_5",
    "actionability_1_5",
    "evidence_trust_1_5",
    "uncertainty_clarity_1_5",
    "missing_perspective",
    "unsupported_claim",
    "overclaim_risk",
    "final_review_status",
]

ENUMS = {
    "missing_perspective": {"yes", "no", "unsure"},
    "unsupported_claim": {"yes", "no", "unsure"},
    "overclaim_risk": {"low", "medium", "high"},
    "final_review_status": {"accepted", "revised", "rejected", "needs_more_evidence"},
}


def build_review_tasks(
    mode: str,
    queries_path: Path | None = None,
    policy: str = "rag",
    project_id: str = "ai_infra_geo_risk",
    write_report: bool = True,
    preserve_existing_reviews: bool = True,
    existing_records_path: Path | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    resolved_queries_path = resolve_path(settings.root, queries_path or DEFAULT_QUERY_PATH)
    output_path = settings.reports_dir / f"intelligence_brief_review_tasks_{mode}.json"
    existing = existing_reviews_by_id(existing_records_path or output_path, preserve_existing_reviews)
    queries = load_queries(resolved_queries_path)
    records = []
    for query in queries:
        brief_report = run_intelligence_brief(
            question=query["query"],
            mode=mode,
            policy=policy,
            project_id=project_id,
            write_report=False,
        )
        records.append(
            {
                "id": str(query.get("id") or stable_id(query["query"])),
                "type": query.get("type"),
                "query": query["query"],
                "brief_report": compact_brief_report(brief_report),
                "review": existing.get(str(query.get("id") or stable_id(query["query"])), empty_review_template()),
            }
        )
    packet = {
        "report_id": REVIEW_PACKET_ID,
        "mode": mode,
        "policy": policy,
        "project_id": project_id,
        "queries_path": relative(settings.root, resolved_queries_path),
        "instructions": review_instructions(),
        "human_label_schema": human_label_schema(),
        "records": records,
    }
    if write_report:
        write_json(output_path, packet)
        write_markdown(
            settings.reports_dir / f"intelligence_brief_review_tasks_{mode}.md",
            render_review_packet(packet),
        )
    return packet


def build_review_status_report(
    records_path: Path,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    records = load_records(records_path)
    rows = [review_status_row(record) for record in records]
    validation_errors = review_validation_errors(records)
    reviewed_count = sum(1 for row in rows if row["complete"])
    report = {
        "report_id": REVIEW_STATUS_ID,
        "records_path": relative(settings.root, resolve_path(settings.root, records_path)),
        "status": "complete" if records and reviewed_count == len(records) and not validation_errors else "incomplete",
        "question_count": len(records),
        "reviewed_question_count": reviewed_count,
        "minimum_question_count": len(records),
        "missing_field_count": sum(len(row["missing_fields"]) for row in rows),
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "rows": rows,
    }
    if write_report:
        write_json(settings.reports_dir / "intelligence_brief_review_status_v1.json", report)
        write_markdown(
            settings.reports_dir / "intelligence_brief_review_status_v1.md",
            render_review_status(report),
        )
    return report


def build_feedback_report(records_path: Path, write_report: bool = True) -> dict[str, Any]:
    settings = load_settings()
    records = load_records(records_path)
    completed = [record for record in records if not missing_review_fields(record) and not validate_review_record(record)]
    report = {
        "report_id": "intelligence_brief_feedback_v1",
        "records_path": relative(settings.root, resolve_path(settings.root, records_path)),
        "status": "ready" if completed else "needs_review_labels",
        "record_count": len(records),
        "reviewed_record_count": len(completed),
        "metrics": feedback_metrics(completed),
        "issue_counts": feedback_issue_counts(completed),
        "packet_metadata_summary": packet_metadata_summary(completed),
        "regression_candidates": regression_candidates(completed),
        "human_review_boundary": (
            "This report summarizes reviewer labels and proposes regression candidates. "
            "It does not change acceptance status or create qrels automatically."
        ),
    }
    if write_report:
        write_json(settings.reports_dir / "intelligence_brief_feedback_v1.json", report)
        write_markdown(
            settings.reports_dir / "intelligence_brief_feedback_v1.md",
            render_feedback_report(report),
        )
    return report


def export_review_csv(records_path: Path, output_path: Path | None = None) -> Path:
    records = load_records(records_path)
    output = output_path or records_path.with_suffix(".review.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(review_csv_row(record))
    return output


def import_review_csv(records_path: Path, csv_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    payload = load_records_payload(records_path)
    records = list(payload.get("records") or [])
    rows = load_review_csv(csv_path)
    known_ids = {str(record.get("id")) for record in records if record.get("id")}
    errors = review_csv_id_errors(rows, known_ids)
    rows_by_id = {str(row.get("id")): row for row in rows if row.get("id")}
    updated = 0
    for record in records:
        row = rows_by_id.get(str(record.get("id")))
        if not row:
            continue
        try:
            record["review"] = review_from_csv_row(row)
        except ValueError as exc:
            errors.append(
                {
                    "id": record.get("id"),
                    "field": "review_csv",
                    "message": str(exc),
                    "value": None,
                }
            )
            continue
        updated += 1
    payload["records"] = records
    errors.extend(review_validation_errors(records))
    if errors:
        return {
            "status": "failed_validation",
            "records_path": str(records_path),
            "csv_path": str(csv_path),
            "records_updated": updated,
            "record_count": len(records),
            "validation_errors": errors,
        }
    output = output_path or records_path
    write_json(output, payload)
    return {
        "status": "imported",
        "records_path": str(records_path),
        "csv_path": str(csv_path),
        "output_path": str(output),
        "records_updated": updated,
        "record_count": len(records),
        "validation_errors": [],
    }


def feedback_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "average_usefulness_1_5": average_review_field(records, "usefulness_1_5"),
        "average_actionability_1_5": average_review_field(records, "actionability_1_5"),
        "average_evidence_trust_1_5": average_review_field(records, "evidence_trust_1_5"),
        "average_uncertainty_clarity_1_5": average_review_field(records, "uncertainty_clarity_1_5"),
        "final_review_status_counts": value_counts(records, "final_review_status"),
    }


def feedback_issue_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "missing_perspective_yes": count_review_value(records, "missing_perspective", "yes"),
        "unsupported_claim_yes": count_review_value(records, "unsupported_claim", "yes"),
        "overclaim_risk_medium_or_high": sum(
            1
            for record in records
            if (record.get("review") or {}).get("overclaim_risk") in {"medium", "high"}
        ),
        "needs_more_evidence_or_rejected": sum(
            1
            for record in records
            if (record.get("review") or {}).get("final_review_status") in {"needs_more_evidence", "rejected"}
        ),
    }


def packet_metadata_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_scope_totals: dict[str, int] = {"private_corpus": 0, "external_source": 0}
    packet_contract_status_counts: dict[str, int] = {}
    external_expansion_status_counts: dict[str, int] = {}
    external_expansion_executed_count = 0
    for record in records:
        report = record.get("brief_report") or {}
        for scope, count in (report.get("evidence_scope_summary") or {}).items():
            scope_key = str(scope)
            evidence_scope_totals[scope_key] = (
                evidence_scope_totals.get(scope_key, 0) + safe_int(count)
            )
        packet_status = (
            (report.get("evidence_packet_contract_validation") or {}).get("status") or "missing"
        )
        packet_contract_status_counts[str(packet_status)] = (
            packet_contract_status_counts.get(str(packet_status), 0) + 1
        )
        expansion = report.get("external_expansion") or {}
        expansion_status = expansion.get("status") or "missing"
        external_expansion_status_counts[str(expansion_status)] = (
            external_expansion_status_counts.get(str(expansion_status), 0) + 1
        )
        if bool(expansion.get("executed")):
            external_expansion_executed_count += 1
    return {
        "evidence_scope_totals": evidence_scope_totals,
        "packet_contract_status_counts": packet_contract_status_counts,
        "external_expansion_status_counts": external_expansion_status_counts,
        "external_expansion_executed_count": external_expansion_executed_count,
    }


def regression_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        review = record.get("review") or {}
        report = record.get("brief_report") or {}
        reasons = []
        if review.get("unsupported_claim") == "yes":
            reasons.append("unsupported_claim")
        if review.get("missing_perspective") == "yes":
            reasons.append("missing_perspective")
        if review.get("overclaim_risk") in {"medium", "high"}:
            reasons.append(f"overclaim_risk_{review.get('overclaim_risk')}")
        if review.get("final_review_status") in {"needs_more_evidence", "rejected"}:
            reasons.append(str(review.get("final_review_status")))
        packet_status = (report.get("evidence_packet_contract_validation") or {}).get("status")
        if packet_status not in {None, "pass"}:
            reasons.append("packet_contract_not_passed")
        if bool((report.get("external_expansion") or {}).get("executed")):
            reasons.append("external_expansion_executed")
        if reasons:
            candidates.append(
                {
                    "id": record.get("id"),
                    "query": record.get("query"),
                    "reasons": reasons,
                    "reviewer_notes": review.get("reviewer_notes", ""),
                    "suggested_regression": "Keep this query in the acceptance set and verify the same issue does not recur.",
                }
            )
    return candidates


def average_review_field(records: list[dict[str, Any]], field: str) -> float | None:
    values = [
        (record.get("review") or {}).get(field)
        for record in records
        if isinstance((record.get("review") or {}).get(field), int)
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def value_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = (record.get("review") or {}).get(field)
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def count_review_value(records: list[dict[str, Any]], field: str, expected: str) -> int:
    return sum(1 for record in records if (record.get("review") or {}).get(field) == expected)


def safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def compact_brief_report(report: dict[str, Any]) -> dict[str, Any]:
    brief = report.get("brief") or {}
    return {
        "status": report.get("status"),
        "question": report.get("question"),
        "evidence_packet_request_id": report.get("evidence_packet_request_id"),
        "evidence_packet_contract_validation": report.get("evidence_packet_contract_validation", {}),
        "evidence_scope_summary": report.get("evidence_scope_summary", {}),
        "external_expansion": report.get("external_expansion", {}),
        "grounding_report": report.get("grounding_report", {}),
        "duplicate_agent_output": report.get("duplicate_agent_output", {}),
        "red_team": report.get("red_team", {}),
        "report_quality": report.get("report_quality", {}),
        "brief": {
            "executive_summary": brief.get("executive_summary"),
            "what_changed": brief.get("what_changed", []),
            "source_gaps": brief.get("source_gaps", []),
            "contradictions_and_uncertainty": brief.get("contradictions_and_uncertainty"),
            "next_signals_to_watch": brief.get("next_signals_to_watch", []),
            "citation_appendix": brief.get("citation_appendix", []),
        },
    }


def review_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    review = record.get("review") or {}
    report = record.get("brief_report") or {}
    brief = report.get("brief") or {}
    grounding = report.get("grounding_report") or {}
    duplicate = report.get("duplicate_agent_output") or {}
    return {
        "id": record.get("id"),
        "type": record.get("type"),
        "query": record.get("query"),
        "brief_status": report.get("status"),
        "executive_summary": brief.get("executive_summary"),
        "what_changed": format_list(brief.get("what_changed") or []),
        "source_gaps": format_json_list(brief.get("source_gaps") or []),
        "red_team_objections": format_json_list((report.get("red_team") or {}).get("objections") or []),
        "citation_appendix": format_citations(brief.get("citation_appendix") or []),
        "evidence_scope_summary": json.dumps(report.get("evidence_scope_summary") or {}, ensure_ascii=False),
        "packet_contract_status": (report.get("evidence_packet_contract_validation") or {}).get("status"),
        "external_expansion_status": (report.get("external_expansion") or {}).get("status"),
        "external_expansion_executed": (report.get("external_expansion") or {}).get("executed"),
        "grounded_claim_ratio": grounding.get("grounded_claim_ratio"),
        "duplicate_agent_output_rate": duplicate.get("duplicate_agent_output_rate"),
        "completed_agents": completed_agents_from_report(report),
        "usefulness_1_5": review.get("usefulness_1_5"),
        "actionability_1_5": review.get("actionability_1_5"),
        "evidence_trust_1_5": review.get("evidence_trust_1_5"),
        "uncertainty_clarity_1_5": review.get("uncertainty_clarity_1_5"),
        "missing_perspective": review.get("missing_perspective"),
        "unsupported_claim": review.get("unsupported_claim"),
        "overclaim_risk": review.get("overclaim_risk"),
        "final_review_status": review.get("final_review_status"),
        "reviewer_notes": review.get("reviewer_notes", ""),
    }


def review_from_csv_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "usefulness_1_5": parse_optional_int(row.get("usefulness_1_5")),
        "actionability_1_5": parse_optional_int(row.get("actionability_1_5")),
        "evidence_trust_1_5": parse_optional_int(row.get("evidence_trust_1_5")),
        "uncertainty_clarity_1_5": parse_optional_int(row.get("uncertainty_clarity_1_5")),
        "missing_perspective": parse_optional_text(row.get("missing_perspective")),
        "unsupported_claim": parse_optional_text(row.get("unsupported_claim")),
        "overclaim_risk": parse_optional_text(row.get("overclaim_risk")),
        "final_review_status": parse_optional_text(row.get("final_review_status")),
        "reviewer_notes": row.get("reviewer_notes") or "",
    }


def review_status_row(record: dict[str, Any]) -> dict[str, Any]:
    missing = missing_review_fields(record)
    errors = validate_review_record(record)
    return {
        "id": record.get("id"),
        "query": record.get("query"),
        "complete": not missing and not errors,
        "missing_fields": missing,
        "error_count": len(errors),
        "final_review_status": (record.get("review") or {}).get("final_review_status"),
    }


def missing_review_fields(record: dict[str, Any]) -> list[str]:
    review = record.get("review") or {}
    missing = []
    for field in REQUIRED_REVIEW_FIELDS:
        value = review.get(field)
        if value is None or value == "" or value == []:
            missing.append(field)
    return missing


def review_validation_errors(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = []
    for record in records:
        for error in validate_review_record(record):
            errors.append({"id": record.get("id"), **error})
    return errors


def validate_review_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    review = record.get("review") or {}
    errors = []
    for field in ["usefulness_1_5", "actionability_1_5", "evidence_trust_1_5", "uncertainty_clarity_1_5"]:
        value = review.get(field)
        if value is None or value == "":
            continue
        if not isinstance(value, int) or not 1 <= value <= 5:
            errors.append({"field": field, "message": "must be an integer from 1 to 5", "value": value})
    for field, allowed in ENUMS.items():
        value = review.get(field)
        if value is None or value == "":
            continue
        if value not in allowed:
            errors.append({"field": field, "message": f"must be one of {sorted(allowed)}", "value": value})
    return errors


def load_records(path: Path) -> list[dict[str, Any]]:
    return list(load_records_payload(path).get("records") or [])


def load_records_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"records": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"records": payload}
    return {"records": []}


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return [row for row in payload if isinstance(row, dict) and row.get("query")]


def load_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def review_csv_id_errors(rows: list[dict[str, str]], known_ids: set[str]) -> list[dict[str, Any]]:
    errors = []
    seen = set()
    for index, row in enumerate(rows, start=2):
        identifier = str(row.get("id") or "").strip()
        if not identifier:
            errors.append({"id": None, "field": "id", "message": f"missing review id on CSV row {index}", "value": row.get("id")})
            continue
        if identifier in seen:
            errors.append({"id": identifier, "field": "id", "message": "duplicate review id in CSV", "value": identifier})
        seen.add(identifier)
        if identifier not in known_ids:
            errors.append({"id": identifier, "field": "id", "message": "CSV review id does not exist in the review packet", "value": identifier})
    return errors


def existing_reviews_by_id(path: Path, preserve: bool) -> dict[str, dict[str, Any]]:
    if not preserve or not path.exists():
        return {}
    return {
        str(record.get("id")): record.get("review")
        for record in load_records(path)
        if record.get("id") and record.get("review")
    }


def empty_review_template() -> dict[str, Any]:
    return {
        "usefulness_1_5": None,
        "actionability_1_5": None,
        "evidence_trust_1_5": None,
        "uncertainty_clarity_1_5": None,
        "missing_perspective": None,
        "unsupported_claim": None,
        "overclaim_risk": None,
        "final_review_status": None,
        "reviewer_notes": "",
    }


def human_label_schema() -> dict[str, Any]:
    return {
        "numeric_1_5": ["usefulness_1_5", "actionability_1_5", "evidence_trust_1_5", "uncertainty_clarity_1_5"],
        "enums": {key: sorted(values) for key, values in ENUMS.items()},
        "required_for_completion": REQUIRED_REVIEW_FIELDS,
    }


def review_instructions() -> list[str]:
    return [
        "Review the brief against the cited evidence, not against general knowledge.",
        "Mark unsupported_claim=yes if any factual claim lacks adequate citation support.",
        "Use needs_more_evidence when the brief is structurally grounded but incomplete for the question.",
        "Judge labels are human-owned; automated gates only prepare the task packet.",
    ]


def format_list(values: list[Any]) -> str:
    return "\n".join(str(value) for value in values)


def format_json_list(values: list[Any]) -> str:
    return "\n".join(json.dumps(value, ensure_ascii=False) for value in values)


def format_citations(values: list[dict[str, Any]]) -> str:
    return "\n".join(
        " | ".join(str(piece) for piece in [item.get("evidence_id"), item.get("title"), item.get("source_name")] if piece)
        for item in values
    )


def completed_agents_from_report(report: dict[str, Any]) -> int:
    quality = report.get("report_quality") or {}
    for check in quality.get("checks") or []:
        if check.get("id") == "minimum_agent_contribution":
            return int((check.get("details") or {}).get("completed_agents") or 0)
    return 0


def parse_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_optional_int(value: str | None) -> int | None:
    text = parse_optional_text(value)
    if text is None:
        return None
    numeric = float(text)
    if int(numeric) != numeric:
        raise ValueError(f"expected integer value, got {value}")
    return int(numeric)


def stable_id(query: str) -> str:
    slug = "".join(character if character.isalnum() else "-" for character in query.lower())[:40].strip("-")
    return slug or "query"


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def render_review_packet(packet: dict[str, Any]) -> str:
    rows = "\n".join(f"- `{record['id']}`: {record['query']}" for record in packet["records"])
    return f"""# Intelligence Brief Review Tasks

Mode: `{packet['mode']}`

Records: {len(packet['records'])}

{rows}
"""


def render_review_status(report: dict[str, Any]) -> str:
    rows = "\n".join(
        f"- `{row['id']}`: {'complete' if row['complete'] else 'missing'} ({', '.join(row['missing_fields'])})"
        for row in report["rows"]
    )
    return f"""# Intelligence Brief Review Status

Status: **{report['status']}**

Reviewed: {report['reviewed_question_count']} / {report['question_count']}

Missing fields: {report['missing_field_count']}

{rows}
"""


def render_feedback_report(report: dict[str, Any]) -> str:
    status_counts = "\n".join(
        f"- `{status}`: {count}"
        for status, count in report["metrics"]["final_review_status_counts"].items()
    ) or "- None"
    issues = "\n".join(
        f"- `{key}`: {value}"
        for key, value in report["issue_counts"].items()
    )
    regressions = "\n".join(
        f"- `{item['id']}`: {', '.join(item['reasons'])} - {item['query']}"
        for item in report["regression_candidates"]
    ) or "- None"
    packet_metadata = report.get("packet_metadata_summary") or {}
    evidence_scope_totals = render_counts(packet_metadata.get("evidence_scope_totals") or {})
    packet_contract_status_counts = render_counts(
        packet_metadata.get("packet_contract_status_counts") or {}
    )
    external_expansion_status_counts = render_counts(
        packet_metadata.get("external_expansion_status_counts") or {}
    )
    return f"""# Intelligence Brief Human Feedback

Status: **{report['status']}**

Reviewed records: {report['reviewed_record_count']} / {report['record_count']}

## Averages

- usefulness: {report['metrics']['average_usefulness_1_5']}
- actionability: {report['metrics']['average_actionability_1_5']}
- evidence trust: {report['metrics']['average_evidence_trust_1_5']}
- uncertainty clarity: {report['metrics']['average_uncertainty_clarity_1_5']}

## Final Status Counts

{status_counts}

## Issue Counts

{issues}

## Packet Metadata

Evidence scope totals:

{evidence_scope_totals}

Packet contract status:

{packet_contract_status_counts}

External expansion status:

{external_expansion_status_counts}

External expansion executed count: {packet_metadata.get('external_expansion_executed_count', 0)}

## Regression Candidates

{regressions}

## Boundary

{report['human_review_boundary']}
"""


def render_counts(counts: dict[str, Any]) -> str:
    return "\n".join(f"- `{key}`: {value}" for key, value in counts.items()) or "- None"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and score human review packets for intelligence briefs.")
    parser.add_argument("--mode", default="ai_infra_geo_risk_demo")
    parser.add_argument("--queries-path", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--policy", default="rag")
    parser.add_argument("--project-id", default="ai_infra_geo_risk")
    parser.add_argument("--records", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-review", action="store_true")
    parser.add_argument("--review-status", action="store_true")
    parser.add_argument("--feedback-report", action="store_true")
    parser.add_argument("--export-review-csv", action="store_true")
    parser.add_argument("--import-review-csv", type=Path)
    parser.add_argument("--reset-review-labels", action="store_true")
    args = parser.parse_args()

    if args.prepare_review:
        packet = build_review_tasks(
            mode=args.mode,
            queries_path=args.queries_path,
            policy=args.policy,
            project_id=args.project_id,
            preserve_existing_reviews=not args.reset_review_labels,
            existing_records_path=args.records,
        )
        print(json.dumps({"status": "review_tasks_written", "record_count": len(packet["records"])}, indent=2))
        return
    if args.review_status:
        if not args.records:
            raise SystemExit("--records is required with --review-status")
        print(json.dumps(build_review_status_report(args.records), indent=2))
        return
    if args.feedback_report:
        if not args.records:
            raise SystemExit("--records is required with --feedback-report")
        print(json.dumps(build_feedback_report(args.records), indent=2))
        return
    if args.export_review_csv:
        if not args.records:
            raise SystemExit("--records is required with --export-review-csv")
        output = export_review_csv(args.records, args.output)
        print(json.dumps({"status": "review_csv_written", "output_path": str(output)}, indent=2))
        return
    if args.import_review_csv:
        if not args.records:
            raise SystemExit("--records is required with --import-review-csv")
        print(json.dumps(import_review_csv(args.records, args.import_review_csv, args.output), indent=2))
        return
    raise SystemExit("Choose --prepare-review, --review-status, --feedback-report, --export-review-csv, or --import-review-csv.")


if __name__ == "__main__":
    main()
