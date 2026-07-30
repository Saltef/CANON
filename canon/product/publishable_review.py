from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import report_io
from canon.product.automated_benchmark_suite import resolve_path
from canon.product.human_review_scaffold import (
    DEFAULT_RETRIEVAL_TARGET,
    DEFAULT_SYNTHESIS_TARGET,
    RETRIEVAL_REVIEW_FIELDS,
    SYNTHESIS_REVIEW_FIELDS,
)


DEFAULT_SCAFFOLD_PATH = Path("reports") / "publishable_human_review_scaffold_v1.json"
DEFAULT_RETRIEVAL_CSV = Path("reports") / "publishable_retrieval_review_scaffold.csv"
DEFAULT_SYNTHESIS_CSV = Path("reports") / "publishable_synthesis_review_scaffold.csv"
DEFAULT_STATUS_PATH = Path("reports") / "publishable_human_review_status_v1.json"

RETRIEVAL_RELEVANCE_VALUES = {"0", "1", "2"}
RETRIEVAL_EVIDENCE_ROLES = {
    "background",
    "supports",
    "contradicts",
    "qualifies",
    "methods",
    "context",
}
SYNTHESIS_ENUMS = {
    "citation_valid": {"yes", "no"},
    "stance_correct": {"yes", "no", "partial"},
    "claim_supported": {"yes", "no", "partial"},
    "missed_key_evidence": {"yes", "no"},
    "reviewer_action": {"accept", "revise", "reject"},
}


def build_publishable_review_status(
    retrieval_csv: Path | None = None,
    synthesis_csv: Path | None = None,
    scaffold_path: Path | None = None,
    output_path: Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    resolved_scaffold = resolve_path(settings.root, scaffold_path or DEFAULT_SCAFFOLD_PATH)
    scaffold = load_scaffold(resolved_scaffold)
    resolved_retrieval_csv = resolve_path(settings.root, retrieval_csv or DEFAULT_RETRIEVAL_CSV)
    resolved_synthesis_csv = resolve_path(settings.root, synthesis_csv or DEFAULT_SYNTHESIS_CSV)
    retrieval_rows = read_csv(resolved_retrieval_csv)
    synthesis_rows = read_csv(resolved_synthesis_csv)
    retrieval = retrieval_status(
        retrieval_rows,
        target=int(
            (scaffold.get("targets") or {}).get("minimum_public_retrieval_queries")
            or DEFAULT_RETRIEVAL_TARGET
        ),
    )
    synthesis = synthesis_status(
        synthesis_rows,
        target=int(
            (scaffold.get("targets") or {}).get("minimum_synthesis_review_questions")
            or DEFAULT_SYNTHESIS_TARGET
        ),
    )
    validation_errors = retrieval["validation_errors"] + synthesis["validation_errors"]
    status = "complete" if retrieval["target_met"] and synthesis["target_met"] and not validation_errors else "incomplete"
    report = {
        "report_id": "publishable_human_review_status_v1",
        "suite_id": scaffold.get("suite_id"),
        "scaffold_path": relative(settings.root, resolved_scaffold),
        "retrieval_csv": relative(settings.root, resolved_retrieval_csv),
        "synthesis_csv": relative(settings.root, resolved_synthesis_csv),
        "status": status,
        "retrieval_review": without_errors(retrieval),
        "synthesis_review": without_errors(synthesis),
        "reviewed_label_count": retrieval["complete_row_count"] + synthesis["complete_row_count"],
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors[:100],
        "truncated_validation_errors": max(0, len(validation_errors) - 100),
        "completion_definition": {
            "retrieval": (
                "A retrieval query is complete when every row for that query has reviewer_id, "
                "valid relevance, and every positively relevant row has a valid evidence_role."
            ),
            "synthesis": (
                "A synthesis case is complete when every row for that case has reviewer_id, "
                "valid citation, stance, support, missed-evidence, and reviewer-action labels."
            ),
            "release_gate": (
                "The publishable package can pass human_review_complete only when both target "
                "counts are met and no validation errors remain."
            ),
        },
        "human_review_boundary": (
            "This report validates reviewer-entered labels. It does not generate labels, "
            "adjudicate disagreements between reviewers, or prove final factual correctness."
        ),
    }
    if write_report:
        output = resolve_path(settings.root, output_path or DEFAULT_STATUS_PATH)
        report_io.write_json(output, report)
        report_io.write_markdown(output.with_suffix(".md"), render_markdown(report))
    return report


def retrieval_status(rows: list[dict[str, str]], target: int) -> dict[str, Any]:
    validation_errors = missing_columns(rows, RETRIEVAL_REVIEW_FIELDS, "retrieval")
    groups: dict[str, list[dict[str, Any]]] = {}
    complete_row_count = 0
    missing_row_count = 0
    started_row_count = 0
    incomplete_examples = []
    for index, row in enumerate(rows, start=2):
        row_errors, complete, started, missing_fields = validate_retrieval_row(row, index)
        validation_errors.extend(row_errors)
        if complete:
            complete_row_count += 1
        else:
            missing_row_count += 1
        if started:
            started_row_count += 1
        if missing_fields and len(incomplete_examples) < 25:
            incomplete_examples.append(
                {
                    "row": index,
                    "query_id": clean(row.get("query_id")),
                    "candidate_chunk_id": clean(row.get("candidate_chunk_id")),
                    "missing_fields": missing_fields,
                }
            )
        query_id = clean(row.get("query_id"))
        if query_id:
            groups.setdefault(query_id, []).append(
                {
                    "complete": complete,
                    "started": started,
                    "validation_error": bool(row_errors),
                }
            )
    complete_queries = [
        query_id
        for query_id, query_rows in groups.items()
        if query_rows and all(row["complete"] and not row["validation_error"] for row in query_rows)
    ]
    started_queries = [
        query_id
        for query_id, query_rows in groups.items()
        if any(row["started"] for row in query_rows)
    ]
    return {
        "target": target,
        "target_met": len(complete_queries) >= target,
        "query_count": len(groups),
        "started_query_count": len(started_queries),
        "complete_query_count": len(complete_queries),
        "row_count": len(rows),
        "started_row_count": started_row_count,
        "complete_row_count": complete_row_count,
        "missing_row_count": missing_row_count,
        "incomplete_examples": incomplete_examples,
        "validation_errors": validation_errors,
    }


def synthesis_status(rows: list[dict[str, str]], target: int) -> dict[str, Any]:
    validation_errors = missing_columns(rows, SYNTHESIS_REVIEW_FIELDS, "synthesis")
    groups: dict[str, list[dict[str, Any]]] = {}
    complete_row_count = 0
    missing_row_count = 0
    started_row_count = 0
    incomplete_examples = []
    for index, row in enumerate(rows, start=2):
        row_errors, complete, started, missing_fields = validate_synthesis_row(row, index)
        validation_errors.extend(row_errors)
        if complete:
            complete_row_count += 1
        else:
            missing_row_count += 1
        if started:
            started_row_count += 1
        if missing_fields and len(incomplete_examples) < 25:
            incomplete_examples.append(
                {
                    "row": index,
                    "case_id": clean(row.get("case_id")),
                    "evidence_id": clean(row.get("evidence_id")),
                    "missing_fields": missing_fields,
                }
            )
        case_id = clean(row.get("case_id"))
        if case_id:
            groups.setdefault(case_id, []).append(
                {
                    "complete": complete,
                    "started": started,
                    "validation_error": bool(row_errors),
                }
            )
    complete_cases = [
        case_id
        for case_id, case_rows in groups.items()
        if case_rows and all(row["complete"] and not row["validation_error"] for row in case_rows)
    ]
    started_cases = [
        case_id
        for case_id, case_rows in groups.items()
        if any(row["started"] for row in case_rows)
    ]
    return {
        "target": target,
        "target_met": len(complete_cases) >= target,
        "case_count": len(groups),
        "started_case_count": len(started_cases),
        "complete_case_count": len(complete_cases),
        "row_count": len(rows),
        "started_row_count": started_row_count,
        "complete_row_count": complete_row_count,
        "missing_row_count": missing_row_count,
        "incomplete_examples": incomplete_examples,
        "validation_errors": validation_errors,
    }


def validate_retrieval_row(row: dict[str, str], row_number: int) -> tuple[list[dict[str, Any]], bool, bool, list[str]]:
    errors = []
    missing = []
    for field in ["query_id", "candidate_chunk_id"]:
        if not clean(row.get(field)):
            errors.append(error("retrieval", row_number, field, f"{field} is required."))
    relevance = clean(row.get("relevance"))
    role = clean(row.get("evidence_role")).lower()
    reviewer_id = clean(row.get("reviewer_id"))
    started = bool(reviewer_id or relevance or role)
    if not reviewer_id:
        missing.append("reviewer_id")
    if not relevance:
        missing.append("relevance")
    elif relevance not in RETRIEVAL_RELEVANCE_VALUES:
        errors.append(error("retrieval", row_number, "relevance", "relevance must be 0, 1, or 2."))
    if relevance and relevance != "0":
        if not role:
            missing.append("evidence_role")
        elif role not in RETRIEVAL_EVIDENCE_ROLES:
            errors.append(
                error(
                    "retrieval",
                    row_number,
                    "evidence_role",
                    f"evidence_role must be one of {sorted(RETRIEVAL_EVIDENCE_ROLES)}.",
                )
            )
    elif role and role not in RETRIEVAL_EVIDENCE_ROLES:
        errors.append(
            error(
                "retrieval",
                row_number,
                "evidence_role",
                f"evidence_role must be blank or one of {sorted(RETRIEVAL_EVIDENCE_ROLES)}.",
            )
        )
    complete = not errors and not missing
    return errors, complete, started, missing


def validate_synthesis_row(row: dict[str, str], row_number: int) -> tuple[list[dict[str, Any]], bool, bool, list[str]]:
    errors = []
    missing = []
    for field in ["case_id", "evidence_id"]:
        if not clean(row.get(field)):
            errors.append(error("synthesis", row_number, field, f"{field} is required."))
    reviewer_id = clean(row.get("reviewer_id"))
    started = bool(reviewer_id or any(clean(row.get(field)) for field in SYNTHESIS_ENUMS))
    if not reviewer_id:
        missing.append("reviewer_id")
    for field, values in SYNTHESIS_ENUMS.items():
        value = clean(row.get(field)).lower()
        if not value:
            missing.append(field)
        elif value not in values:
            errors.append(error("synthesis", row_number, field, f"{field} must be one of {sorted(values)}."))
    reason = clean(row.get("reason"))
    if problem_label_requires_reason(row) and not reason:
        missing.append("reason")
    complete = not errors and not missing
    return errors, complete, started, missing


def problem_label_requires_reason(row: dict[str, str]) -> bool:
    return (
        clean(row.get("citation_valid")).lower() == "no"
        or clean(row.get("stance_correct")).lower() in {"no", "partial"}
        or clean(row.get("claim_supported")).lower() in {"no", "partial"}
        or clean(row.get("missed_key_evidence")).lower() == "yes"
        or clean(row.get("reviewer_action")).lower() in {"revise", "reject"}
    )


def missing_columns(rows: list[dict[str, str]], expected: list[str], review_set: str) -> list[dict[str, Any]]:
    if not rows:
        return [error(review_set, 1, "csv", "CSV contains no review rows.")]
    actual = set(rows[0])
    return [
        error(review_set, 1, field, f"CSV is missing required column {field}.")
        for field in expected
        if field not in actual
    ]


def without_errors(section: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in section.items() if key != "validation_errors"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Review CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_scaffold(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "suite_id": None,
            "targets": {
                "minimum_public_retrieval_queries": DEFAULT_RETRIEVAL_TARGET,
                "minimum_synthesis_review_questions": DEFAULT_SYNTHESIS_TARGET,
            },
        }
    return json.loads(path.read_text(encoding="utf-8"))


def error(review_set: str, row_number: int, field: str, message: str) -> dict[str, Any]:
    return {
        "review_set": review_set,
        "row": row_number,
        "field": field,
        "message": message,
    }


def clean(value: Any) -> str:
    return str(value or "").strip()


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def render_markdown(report: dict[str, Any]) -> str:
    retrieval = report["retrieval_review"]
    synthesis = report["synthesis_review"]
    return f"""# CANON Publishable Human Review Status

Status: `{report['status']}`

Suite: `{report.get('suite_id')}`

Retrieval complete queries: `{retrieval['complete_query_count']}` / `{retrieval['target']}`

Synthesis complete cases: `{synthesis['complete_case_count']}` / `{synthesis['target']}`

Reviewed label rows: `{report['reviewed_label_count']}`

Validation errors: `{report['validation_error_count']}`

This status validates reviewer-entered labels for the publishable package. It
does not create labels or remove the need for human adjudication.
"""


def cli_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": report["report_id"],
        "suite_id": report.get("suite_id"),
        "status": report["status"],
        "retrieval": {
            "complete_query_count": report["retrieval_review"]["complete_query_count"],
            "target": report["retrieval_review"]["target"],
            "target_met": report["retrieval_review"]["target_met"],
        },
        "synthesis": {
            "complete_case_count": report["synthesis_review"]["complete_case_count"],
            "target": report["synthesis_review"]["target"],
            "target_met": report["synthesis_review"]["target_met"],
        },
        "validation_error_count": report["validation_error_count"],
        "artifacts": {
            "json": "reports/publishable_human_review_status_v1.json",
            "markdown": "reports/publishable_human_review_status_v1.md",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate completed publishable review labels.")
    parser.add_argument("--retrieval-csv", default=str(DEFAULT_RETRIEVAL_CSV))
    parser.add_argument("--synthesis-csv", default=str(DEFAULT_SYNTHESIS_CSV))
    parser.add_argument("--scaffold", default=str(DEFAULT_SCAFFOLD_PATH))
    parser.add_argument("--output", default=str(DEFAULT_STATUS_PATH))
    args = parser.parse_args()
    report = build_publishable_review_status(
        retrieval_csv=Path(args.retrieval_csv),
        synthesis_csv=Path(args.synthesis_csv),
        scaffold_path=Path(args.scaffold),
        output_path=Path(args.output),
    )
    print(json.dumps(cli_summary(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
