from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import report_io
from canon.product import service


MIN_QUESTION_COUNT = 30
REVIEW_CSV_FIELDS = [
    "id",
    "category",
    "query",
    "answer",
    "citations",
    "evidence_previews",
    "support_assessment",
    "limitations",
    "conflict_notes",
    "matched_terms",
    "weak_terms",
    "field_phrases",
    "query_variants",
    "diagnostic_stability",
    "relevant_evidence_count",
    "substantive_claim_count",
    "unsupported_claim_count",
    "brief_review_minutes",
    "evidence_relevance",
    "citation_support",
    "answer_usefulness",
    "query_lingo_ratings",
    "drift_risk_review",
    "final_review_status",
    "reviewer_notes",
]


ACCEPTANCE_QUESTIONS = [
    {
        "id": "q01",
        "category": "straightforward_evidence",
        "query": "What does the literature say about whether democracies are less likely to go to war with each other?",
    },
    {
        "id": "q02",
        "category": "straightforward_evidence",
        "query": "How effective are economic sanctions at changing state behavior?",
    },
    {
        "id": "q03",
        "category": "straightforward_evidence",
        "query": "Do international human rights treaties improve state human rights practices?",
    },
    {
        "id": "q04",
        "category": "straightforward_evidence",
        "query": "What methods dominate research on civil war onset, and what are their main critiques?",
    },
    {
        "id": "q05",
        "category": "straightforward_evidence",
        "query": "How do institutional voids affect social entrepreneurship and international business?",
    },
    {
        "id": "q06",
        "category": "vague_beginner_wording",
        "query": "Do democracies fight each other?",
    },
    {
        "id": "q07",
        "category": "vague_beginner_wording",
        "query": "Do sanctions work?",
    },
    {
        "id": "q08",
        "category": "vague_beginner_wording",
        "query": "Do human rights treaties actually help people?",
    },
    {
        "id": "q09",
        "category": "vague_beginner_wording",
        "query": "Why do civil wars start again after peace?",
    },
    {
        "id": "q10",
        "category": "vague_beginner_wording",
        "query": "Do international organizations really change what states do?",
    },
    {
        "id": "q11",
        "category": "terminology_sensitive",
        "query": "What field vocabulary should a beginner use when searching democratic peace theory?",
    },
    {
        "id": "q12",
        "category": "terminology_sensitive",
        "query": "What is the difference between sanction imposition and sanction threat?",
    },
    {
        "id": "q13",
        "category": "terminology_sensitive",
        "query": "Which terms describe collateral effects of sanctions on civilians?",
    },
    {
        "id": "q14",
        "category": "terminology_sensitive",
        "query": "What phrases do researchers use for ethnic exclusion and transborder kin in civil war studies?",
    },
    {
        "id": "q15",
        "category": "terminology_sensitive",
        "query": "What vocabulary should a newcomer use when searching naming and shaming mechanisms in human rights research?",
    },
    {
        "id": "q16",
        "category": "mixed_evidence",
        "query": "Where does the evidence disagree about democratic peace findings?",
    },
    {
        "id": "q17",
        "category": "mixed_evidence",
        "query": "What are the strongest arguments that sanctions succeed, and what are the strongest criticisms?",
    },
    {
        "id": "q18",
        "category": "mixed_evidence",
        "query": "Where do civil war studies disagree about grievances, resources, and recurrence?",
    },
    {
        "id": "q19",
        "category": "mixed_evidence",
        "query": "Map the main disagreements about whether international institutions independently affect state behavior.",
    },
    {
        "id": "q20",
        "category": "mixed_evidence",
        "query": "What does the corpus say about climate change and the Syrian civil war without overgeneralizing?",
    },
    {
        "id": "q21",
        "category": "weak_support",
        "query": "Can this corpus prove that democracies never fight covert or indirect wars?",
    },
    {
        "id": "q22",
        "category": "weak_support",
        "query": "Is there enough evidence here to recommend sanctions as a policy in every case?",
    },
    {
        "id": "q23",
        "category": "weak_support",
        "query": "Can this corpus rank every country by civil war risk?",
    },
    {
        "id": "q24",
        "category": "recency_sensitive",
        "query": "Which post-2020 works in the corpus should affect conclusions most?",
    },
    {
        "id": "q25",
        "category": "recency_sensitive",
        "query": "What newer research changes conclusions about UN treaty bodies and human rights compliance?",
    },
    {
        "id": "q26",
        "category": "source_quality_sensitive",
        "query": "Which retrieved sources look strongest for a cautious briefing on institutions and cooperation?",
    },
    {
        "id": "q27",
        "category": "source_quality_sensitive",
        "query": "Which retrieved sources should be treated carefully because of weak support, narrow scope, or repeated-source concentration?",
    },
    {
        "id": "q28",
        "category": "overclaim_check",
        "query": "Overclaim check: 'Democratic peace theory proves democracies never use force against each other.'",
    },
    {
        "id": "q29",
        "category": "overclaim_check",
        "query": "Overclaim check: 'Economic sanctions are harmless to civilians because they only target leaders.'",
    },
    {
        "id": "q30",
        "category": "unique_use_case",
        "query": "Teach a newcomer the search vocabulary across sanctions, democratic peace, civil war, institutions, and human rights: which terms should they try next, and which terms look too vague?",
    },
]


def build_industry_pilot_report(
    mode: str = service.DEFAULT_MODE,
    records_path: Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    records = load_records(records_path) if records_path else []
    validation_errors = review_validation_errors(records)
    checks = acceptance_checks(records)
    checks.insert(
        0,
        check(
            "review_labels_valid",
            not validation_errors,
            "Human review labels must use the expected schema and numeric ranges.",
            len(validation_errors),
        ),
    )
    report = {
        "report_id": "industry_pilot_acceptance_v1",
        "mode": mode,
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "question_count": len(records),
        "reviewed_question_count": reviewed_question_count(records),
        "minimum_question_count": MIN_QUESTION_COUNT,
        "checks": checks,
        "release_blockers": release_blockers(checks),
        "required_artifacts": required_artifacts(mode),
        "human_label_schema": human_label_schema(),
        "review_validation_errors": validation_errors,
        "records": compact_records(records),
    }
    if write_report:
        write_json(settings.reports_dir / "industry_pilot_acceptance_v1.json", report)
        write_markdown(settings.reports_dir / "industry_pilot_acceptance_v1.md", render_markdown(report))
    return report


def build_review_status_report(
    mode: str = service.DEFAULT_MODE,
    records_path: Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    records = load_records(records_path) if records_path else []
    rows = [review_status_row(record) for record in records]
    validation_errors = review_validation_errors(records)
    complete_count = sum(1 for row in rows if row["complete"])
    report = {
        "report_id": "human_review_status_v1",
        "mode": mode,
        "status": "complete" if complete_count >= MIN_QUESTION_COUNT else "incomplete",
        "question_count": len(records),
        "reviewed_question_count": complete_count,
        "minimum_question_count": MIN_QUESTION_COUNT,
        "missing_field_count": sum(len(row["missing_fields"]) for row in rows),
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "rows": rows,
    }
    if write_report:
        write_json(settings.reports_dir / "human_review_status_v1.json", report)
        write_markdown(settings.reports_dir / "human_review_status_v1.md", render_review_status(report))
    return report


def export_review_csv(
    records_path: Path,
    output_path: Path | None = None,
) -> Path:
    records = load_records(records_path)
    output = output_path or records_path.with_suffix(".review.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(review_csv_row(record))
    return output


def import_review_csv(
    records_path: Path,
    csv_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    payload = load_records_payload(records_path)
    records = list(payload.get("records") or [])
    rows = load_review_csv(csv_path)
    known_ids = {str(record.get("id")) for record in records if record.get("id")}
    csv_id_errors = review_csv_id_errors(rows, known_ids)
    rows_by_id = {row["id"]: row for row in rows if row.get("id")}
    updated = 0
    parse_errors = []
    for record in records:
        row = rows_by_id.get(str(record.get("id")))
        if not row:
            continue
        try:
            record["review"] = review_from_csv_row(row)
        except ValueError as exc:
            parse_errors.append(
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
    validation_errors = csv_id_errors + parse_errors + review_validation_errors(records)
    if validation_errors:
        return {
            "status": "failed_validation",
            "records_path": str(records_path),
            "csv_path": str(csv_path),
            "records_updated": updated,
            "record_count": len(records),
            "validation_errors": validation_errors,
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


def build_human_review_tasks(
    mode: str = service.DEFAULT_MODE,
    questions: list[dict[str, Any]] | None = None,
    write_report: bool = True,
    answer_fn=service.answer,
    preserve_existing_reviews: bool = True,
    existing_records_path: Path | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    output_path = settings.reports_dir / "human_review_tasks_v1.json"
    existing_reviews = existing_reviews_by_id(
        existing_records_path or output_path,
        preserve_existing_reviews,
    )
    tasks = []
    for question in questions or ACCEPTANCE_QUESTIONS:
        answer_response = compact_review_answer_response(
            answer_fn(
            {
                "query": question["query"],
                "mode": mode,
                "policy": service.DEFAULT_POLICY,
                "top_k": 10,
                "freedom_level": "exploratory"
                if question["category"] in {"vague_beginner_wording", "unique_use_case"}
                else "balanced",
            }
            )
        )
        tasks.append(
            {
                "id": question["id"],
                "category": question["category"],
                "query": question["query"],
                "answer_response": answer_response,
                "review": existing_reviews.get(question["id"], empty_review_template()),
            }
        )
    packet = {
        "report_id": "human_review_tasks_v1",
        "mode": mode,
        "instructions": review_instructions(),
        "human_label_schema": human_label_schema(),
        "records": tasks,
    }
    if write_report:
        write_json(output_path, packet)
        write_json(
            settings.reports_dir / "query_diagnostics_acceptance_v1.json",
            query_diagnostics_acceptance(tasks, mode),
        )
        write_markdown(
            settings.reports_dir / "evidence_briefing_acceptance_v1.md",
            render_evidence_briefing(tasks, mode),
        )
    return packet


def existing_reviews_by_id(path: Path, preserve: bool) -> dict[str, dict[str, Any]]:
    if not preserve or not path.exists():
        return {}
    return {
        str(record.get("id")): review(record)
        for record in load_records(path)
        if record.get("id") and review(record)
    }


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


def load_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def review_csv_id_errors(rows: list[dict[str, str]], known_ids: set[str]) -> list[dict[str, Any]]:
    errors = []
    seen = set()
    for index, row in enumerate(rows, start=2):
        identifier = str(row.get("id") or "").strip()
        if not identifier:
            errors.append(
                {
                    "id": None,
                    "field": "id",
                    "message": f"missing review id on CSV row {index}",
                    "value": row.get("id"),
                }
            )
            continue
        if identifier in seen:
            errors.append(
                {
                    "id": identifier,
                    "field": "id",
                    "message": "duplicate review id in CSV",
                    "value": identifier,
                }
            )
        seen.add(identifier)
        if identifier not in known_ids:
            errors.append(
                {
                    "id": identifier,
                    "field": "id",
                    "message": "CSV review id does not exist in the review packet",
                    "value": identifier,
                }
            )
    return errors


def review_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    payload = review(record)
    answer_payload = answer(record)
    diagnostics = answer_payload.get("query_diagnostics") or {}
    query_to_corpus = diagnostics.get("query_to_corpus") or {}
    neighborhood = diagnostics.get("result_neighborhood") or {}
    return {
        "id": record.get("id"),
        "category": record.get("category"),
        "query": record.get("query") or answer_payload.get("query"),
        "answer": answer_payload.get("answer"),
        "citations": format_csv_citations(answer_payload.get("citations") or []),
        "evidence_previews": format_csv_evidence(answer_payload.get("evidence") or []),
        "support_assessment": json.dumps(answer_payload.get("support_assessment") or {}, ensure_ascii=False),
        "limitations": format_csv_list(answer_payload.get("limitations") or []),
        "conflict_notes": format_csv_json_list(answer_payload.get("conflict_notes") or []),
        "matched_terms": format_csv_list(query_to_corpus.get("matched_terms") or []),
        "weak_terms": format_csv_list(query_to_corpus.get("weak_terms") or []),
        "field_phrases": format_csv_list(neighborhood.get("field_phrases") or []),
        "query_variants": format_csv_query_variants(diagnostics.get("query_variants") or []),
        "diagnostic_stability": json.dumps(diagnostics.get("stability") or {}, ensure_ascii=False),
        "relevant_evidence_count": payload.get("relevant_evidence_count"),
        "substantive_claim_count": payload.get("substantive_claim_count"),
        "unsupported_claim_count": payload.get("unsupported_claim_count"),
        "brief_review_minutes": payload.get("brief_review_minutes"),
        "evidence_relevance": payload.get("evidence_relevance"),
        "citation_support": payload.get("citation_support"),
        "answer_usefulness": payload.get("answer_usefulness"),
        "query_lingo_ratings": ";".join(payload.get("query_lingo_ratings") or []),
        "drift_risk_review": payload.get("drift_risk_review"),
        "final_review_status": payload.get("final_review_status"),
        "reviewer_notes": payload.get("reviewer_notes", ""),
    }


def format_csv_citations(citations: list[dict[str, Any]]) -> str:
    return "\n".join(
        " | ".join(
            str(piece)
            for piece in [
                citation.get("citation_id"),
                citation.get("title"),
                citation.get("source_name"),
                citation.get("year"),
                citation.get("chunk_id"),
            ]
            if piece is not None
        )
        for citation in citations[:10]
    )


def format_csv_evidence(evidence: list[dict[str, Any]]) -> str:
    rows = []
    for item in evidence[:10]:
        claim = item.get("claim") or {}
        rows.append(
            " | ".join(
                str(piece)
                for piece in [
                    f"rank={item.get('rank')}",
                    item.get("citation_id"),
                    item.get("title"),
                    item.get("source_name"),
                    f"score={item.get('final_score')}",
                    claim.get("text"),
                    item.get("preview"),
                ]
                if piece is not None
            )
        )
    return "\n\n".join(rows)


def format_csv_list(values: list[Any]) -> str:
    return "; ".join(str(value) for value in values)


def format_csv_json_list(values: list[Any]) -> str:
    if not values:
        return ""
    return "\n".join(json.dumps(value, ensure_ascii=False) for value in values)


def format_csv_query_variants(variants: list[dict[str, Any]]) -> str:
    rows = []
    for variant in variants:
        rows.append(
            " | ".join(
                str(piece)
                for piece in [
                    variant.get("query"),
                    f"drift={variant.get('drift_risk')}",
                    f"freedom={variant.get('freedom_level')}",
                    f"overlap={variant.get('result_overlap_with_original')}",
                    variant.get("reason"),
                ]
                if piece is not None
            )
        )
    return "\n".join(rows)


def review_from_csv_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "relevant_evidence_count": parse_optional_int(row.get("relevant_evidence_count")),
        "substantive_claim_count": parse_optional_int(row.get("substantive_claim_count")),
        "unsupported_claim_count": parse_optional_int(row.get("unsupported_claim_count")),
        "brief_review_minutes": parse_optional_float(row.get("brief_review_minutes")),
        "evidence_relevance": parse_optional_text(row.get("evidence_relevance")),
        "citation_support": parse_optional_text(row.get("citation_support")),
        "answer_usefulness": parse_optional_text(row.get("answer_usefulness")),
        "query_lingo_ratings": parse_ratings(row.get("query_lingo_ratings")),
        "drift_risk_review": parse_optional_text(row.get("drift_risk_review")),
        "final_review_status": parse_optional_text(row.get("final_review_status")),
        "reviewer_notes": row.get("reviewer_notes") or "",
    }


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


def parse_optional_float(value: str | None) -> float | None:
    text = parse_optional_text(value)
    if text is None:
        return None
    return float(text)


def parse_ratings(value: str | None) -> list[str]:
    text = parse_optional_text(value)
    if text is None:
        return []
    return [piece.strip() for piece in text.replace(",", ";").split(";") if piece.strip()]


def acceptance_checks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        check(
            "acceptance_question_count",
            reviewed_question_count(records) >= MIN_QUESTION_COUNT,
            f"At least {MIN_QUESTION_COUNT} reviewed acceptance questions are required.",
            reviewed_question_count(records),
        ),
        check(
            "relevant_evidence_top10_rate",
            rate(records, has_three_relevant_items) >= 0.8,
            "At least 80% of questions need three or more reviewer-rated relevant evidence items in top 10.",
            rate(records, has_three_relevant_items),
        ),
        check(
            "brief_required_fields_rate",
            rate(records, answer_has_required_fields) >= 0.9,
            "At least 90% of briefs must include citations, support assessment, limitations, and conflict notes.",
            rate(records, answer_has_required_fields),
        ),
        check(
            "unsupported_claim_rate",
            unsupported_claim_rate(records) < 0.05,
            "Fewer than 5% of substantive answer claims may be reviewer-rated unsupported.",
            unsupported_claim_rate(records),
        ),
        check(
            "query_diagnostics_rate",
            rate(records, answer_has_query_diagnostics) >= 1.0,
            "Every answer must include query diagnostics.",
            rate(records, answer_has_query_diagnostics),
        ),
        check(
            "query_variant_metadata_rate",
            rate(records, query_variants_are_labeled) >= 1.0,
            "Every query variant must include result overlap, drift risk, and exploration level.",
            rate(records, query_variants_are_labeled),
        ),
        check(
            "field_phrase_traceability_rate",
            field_phrase_traceability_rate(records) >= 0.8,
            "At least 80% of suggested field phrases must be traceable to retrieved evidence or accepted reviewer patterns.",
            field_phrase_traceability_rate(records),
        ),
        check(
            "query_lingo_usefulness_rate",
            query_lingo_usefulness_rate(records) >= 0.7,
            "Reviewers must rate at least 70% of field phrase suggestions as useful or reasonable.",
            query_lingo_usefulness_rate(records),
        ),
        check(
            "brief_review_time_rate",
            rate(records, reviewed_within_15_minutes) >= 0.7,
            "At least 70% of first-pass briefs must be reviewed in 15 minutes or less.",
            rate(records, reviewed_within_15_minutes),
        ),
        check(
            "no_silent_query_rewrites",
            all(no_silent_rewrites(record) for record in records) if records else False,
            "Exploratory variants must be visible, optional, and recorded.",
            rate(records, no_silent_rewrites),
        ),
    ]


def check(identifier: str, passed: bool, requirement: str, observed: Any) -> dict[str, Any]:
    return {
        "id": identifier,
        "passed": bool(passed),
        "requirement": requirement,
        "observed": observed,
    }


def rate(records: list[dict[str, Any]], predicate) -> float:
    if not records:
        return 0.0
    return round(sum(1 for record in records if predicate(record)) / len(records), 6)


def answer(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("answer_response") or {}


def review(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("review") or {}


def reviewed_question_count(records: list[dict[str, Any]]) -> int:
    return sum(1 for record in records if review_is_complete(record))


def review_is_complete(record: dict[str, Any]) -> bool:
    return not missing_review_fields(record)


def missing_review_fields(record: dict[str, Any]) -> list[str]:
    payload = review(record)
    required_scalar_fields = [
        "relevant_evidence_count",
        "substantive_claim_count",
        "unsupported_claim_count",
        "brief_review_minutes",
        "evidence_relevance",
        "citation_support",
        "answer_usefulness",
        "drift_risk_review",
        "final_review_status",
    ]
    missing = [field for field in required_scalar_fields if payload.get(field) is None]
    if not payload.get("query_lingo_ratings"):
        missing.append("query_lingo_ratings")
    return missing


def review_status_row(record: dict[str, Any]) -> dict[str, Any]:
    missing = missing_review_fields(record)
    errors = validate_review_record(record)
    return {
        "id": record.get("id"),
        "category": record.get("category"),
        "query": record.get("query") or answer(record).get("query"),
        "complete": not missing and not errors,
        "missing_fields": missing,
        "validation_errors": errors,
        "review_minutes": review(record).get("brief_review_minutes"),
        "final_review_status": review(record).get("final_review_status"),
    }


def review_validation_errors(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = []
    for record in records:
        for error in validate_review_record(record):
            errors.append(
                {
                    "id": record.get("id"),
                    "field": error["field"],
                    "message": error["message"],
                    "value": error.get("value"),
                }
            )
    return errors


def validate_review_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    payload = review(record)
    if not payload:
        return []
    errors = []
    schema = human_label_schema()
    numeric_fields = {
        "relevant_evidence_count": {"min": 0, "integer": True},
        "substantive_claim_count": {"min": 0, "integer": True},
        "unsupported_claim_count": {"min": 0, "integer": True},
        "brief_review_minutes": {"min": 0, "integer": False},
    }
    for field, rules in numeric_fields.items():
        value = payload.get(field)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(error(field, "must be numeric", value))
            continue
        if numeric < rules["min"]:
            errors.append(error(field, f"must be >= {rules['min']}", value))
        if rules["integer"] and int(numeric) != numeric:
            errors.append(error(field, "must be an integer", value))
    if (
        payload.get("unsupported_claim_count") is not None
        and payload.get("substantive_claim_count") is not None
    ):
        unsupported = int(float(payload.get("unsupported_claim_count")))
        substantive = int(float(payload.get("substantive_claim_count")))
        if unsupported > substantive:
            errors.append(
                error(
                    "unsupported_claim_count",
                    "cannot exceed substantive_claim_count",
                    payload.get("unsupported_claim_count"),
                )
            )
    enum_fields = {
        "evidence_relevance": "evidence_relevance",
        "citation_support": "citation_support",
        "answer_usefulness": "answer_usefulness",
        "drift_risk_review": "drift_risk_review",
        "final_review_status": "final_review_status",
    }
    for field, schema_key in enum_fields.items():
        value = payload.get(field)
        if value is not None and value not in schema[schema_key]:
            errors.append(error(field, f"must be one of {schema[schema_key]}", value))
    ratings = payload.get("query_lingo_ratings")
    if ratings:
        if not isinstance(ratings, list):
            errors.append(error("query_lingo_ratings", "must be a list", ratings))
        else:
            allowed = set(schema["query_lingo_usefulness"])
            for rating in ratings:
                if rating not in allowed:
                    errors.append(error("query_lingo_ratings", f"contains invalid rating {rating}", rating))
    return errors


def error(field: str, message: str, value: Any) -> dict[str, Any]:
    return {"field": field, "message": message, "value": value}


def has_three_relevant_items(record: dict[str, Any]) -> bool:
    return int(review(record).get("relevant_evidence_count") or 0) >= 3


def answer_has_required_fields(record: dict[str, Any]) -> bool:
    payload = answer(record)
    return all(
        key in payload
        for key in ["citations", "support_assessment", "limitations", "conflict_notes"]
    )


def unsupported_claim_rate(records: list[dict[str, Any]]) -> float:
    claim_count = sum(int(review(record).get("substantive_claim_count") or 0) for record in records)
    unsupported_count = sum(int(review(record).get("unsupported_claim_count") or 0) for record in records)
    if claim_count <= 0:
        return 1.0
    return round(unsupported_count / claim_count, 6)


def answer_has_query_diagnostics(record: dict[str, Any]) -> bool:
    diagnostics = answer(record).get("query_diagnostics") or {}
    return all(
        key in diagnostics
        for key in ["query_to_corpus", "result_neighborhood", "query_variants", "stability"]
    )


def query_variants_are_labeled(record: dict[str, Any]) -> bool:
    variants = ((answer(record).get("query_diagnostics") or {}).get("query_variants") or [])
    if not variants:
        return True
    return all(
        "result_overlap_with_original" in variant
        and "drift_risk" in variant
        and "exploration_label" in variant
        and "freedom_level" in variant
        for variant in variants
    )


def field_phrase_traceability_rate(records: list[dict[str, Any]]) -> float:
    details = [
        detail
        for record in records
        for detail in (((answer(record).get("query_diagnostics") or {}).get("result_neighborhood") or {}).get("phrase_details") or [])
    ]
    if not details:
        return 0.0
    traceable = [
        detail
        for detail in details
        if detail.get("source") in {"semantic_result_neighborhood", "accepted_reviewer_pattern"}
        and int(detail.get("support_count") or 0) > 0
    ]
    return round(len(traceable) / len(details), 6)


def query_lingo_usefulness_rate(records: list[dict[str, Any]]) -> float:
    ratings = [
        rating
        for record in records
        for rating in list(review(record).get("query_lingo_ratings") or [])
    ]
    if not ratings:
        return 0.0
    useful = [rating for rating in ratings if rating in {"reasonable", "useful", "very_useful"}]
    return round(len(useful) / len(ratings), 6)


def reviewed_within_15_minutes(record: dict[str, Any]) -> bool:
    minutes = review(record).get("brief_review_minutes")
    return minutes is not None and float(minutes) <= 15.0


def no_silent_rewrites(record: dict[str, Any]) -> bool:
    diagnostics = answer(record).get("query_diagnostics") or {}
    if diagnostics.get("original_query") != answer(record).get("query"):
        return False
    for variant in diagnostics.get("query_variants") or []:
        if variant.get("applied_silently"):
            return False
    return True


def release_blockers(checks: list[dict[str, Any]]) -> list[str]:
    return [check["id"] for check in checks if not check["passed"]]


def required_artifacts(mode: str) -> list[str]:
    return [
        "reports/industry_pilot_acceptance_v1.json",
        "reports/industry_pilot_acceptance_v1.md",
        "reports/query_diagnostics_acceptance_v1.json",
        "reports/evidence_briefing_acceptance_v1.md",
        "reports/human_review_tasks_v1.json",
        "reports/human_review_status_v1.json",
        "reports/human_review_status_v1.md",
        "docs/human_review_rubric.md",
        f"reports/product_readiness_{mode}.json",
        f"reports/regression_gate_{mode}_baseline_methods_v1.json",
    ]


def human_label_schema() -> dict[str, list[str]]:
    return {
        "evidence_relevance": ["irrelevant", "partial", "relevant", "highly_relevant"],
        "citation_support": ["unsupported", "weak", "adequate", "strong"],
        "answer_usefulness": ["unusable", "major_revision", "minor_revision", "usable"],
        "query_lingo_usefulness": ["not_useful", "somewhat_useful", "useful", "very_useful"],
        "drift_risk_review": ["acceptable", "needs_caution", "drifted"],
        "final_review_status": ["accepted", "revised", "rejected", "needs_more_evidence"],
    }


def empty_review_template() -> dict[str, Any]:
    return {
        "relevant_evidence_count": None,
        "substantive_claim_count": None,
        "unsupported_claim_count": None,
        "brief_review_minutes": None,
        "evidence_relevance": None,
        "citation_support": None,
        "answer_usefulness": None,
        "query_lingo_ratings": [],
        "drift_risk_review": None,
        "final_review_status": None,
        "reviewer_notes": "",
    }


def review_instructions() -> list[str]:
    return [
        "Use docs/human_review_rubric.md as the labeling standard.",
        "Review each answer as a first-pass analyst assist, not as an autonomous decision.",
        "Count evidence as relevant only when it directly helps answer the user query.",
        "Count substantive claims in the generated answer, then count claims not supported by cited evidence.",
        "Rate query-lingo suggestions after checking whether they would help a newcomer search the topic.",
        "Leave exploratory query variants visible; do not treat them as replacements for the original query.",
        "Run the industry pilot gate against this completed file after all review fields are filled.",
    ]


def query_diagnostics_acceptance(tasks: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    rows = []
    for task in tasks:
        diagnostics = (answer(task).get("query_diagnostics") or {})
        neighborhood = diagnostics.get("result_neighborhood") or {}
        rows.append(
            {
                "id": task.get("id"),
                "category": task.get("category"),
                "query": task.get("query"),
                "stability": diagnostics.get("stability", {}),
                "matched_terms": (diagnostics.get("query_to_corpus") or {}).get("matched_terms", []),
                "weak_terms": (diagnostics.get("query_to_corpus") or {}).get("weak_terms", []),
                "field_phrases": neighborhood.get("field_phrases", []),
                "missing_from_query": neighborhood.get("missing_from_query", []),
                "query_variants": diagnostics.get("query_variants", []),
            }
        )
    return {
        "report_id": "query_diagnostics_acceptance_v1",
        "mode": mode,
        "question_count": len(tasks),
        "rows": rows,
    }


def compact_review_answer_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": payload.get("query"),
        "mode": payload.get("mode"),
        "policy": payload.get("policy"),
        "answer": payload.get("answer"),
        "citations": payload.get("citations", []),
        "evidence": [compact_review_evidence(item) for item in payload.get("evidence", [])],
        "support_assessment": payload.get("support_assessment", {}),
        "limitations": payload.get("limitations", []),
        "conflict_notes": payload.get("conflict_notes", []),
        "claim_boundaries": payload.get("claim_boundaries", {}),
        "query_diagnostics": payload.get("query_diagnostics", {}),
    }


def compact_review_evidence(item: dict[str, Any]) -> dict[str, Any]:
    claim = item.get("claim") or {}
    explanation = item.get("explanation") or {}
    return {
        "citation_id": item.get("citation_id"),
        "chunk_id": item.get("chunk_id"),
        "work_id": item.get("work_id"),
        "title": item.get("title"),
        "source_name": item.get("source_name"),
        "year": item.get("year"),
        "rank": item.get("rank"),
        "cluster_id": item.get("cluster_id"),
        "final_score": item.get("final_score"),
        "claim": {
            "text": claim.get("text"),
            "stance": claim.get("stance"),
            "confidence": claim.get("confidence"),
        }
        if claim
        else None,
        "preview": item.get("preview"),
        "explanation": {
            "summary": explanation.get("summary"),
            "reasons": explanation.get("reasons", []),
        },
        "conflict_injected": bool(item.get("conflict_injected")),
    }


def render_evidence_briefing(tasks: list[dict[str, Any]], mode: str) -> str:
    sections = [
        "# CANON Evidence Briefing Acceptance",
        "",
        f"Mode: `{mode}`",
        "",
        "Use this file for human review. The JSON task packet is the authoritative record for gate scoring.",
        "",
    ]
    for task in tasks:
        payload = answer(task)
        diagnostics = payload.get("query_diagnostics") or {}
        phrases = ((diagnostics.get("result_neighborhood") or {}).get("field_phrases") or [])[:8]
        citations = payload.get("citations") or []
        sections.extend(
            [
                f"## {task.get('id')} - {task.get('category')}",
                "",
                f"Query: {task.get('query')}",
                "",
                "### Answer",
                "",
                str(payload.get("answer") or "").strip(),
                "",
                "### Citations",
                "",
                render_citations(citations),
                "",
                "### Query Lingo",
                "",
                "- Stability: "
                + str((diagnostics.get("stability") or {}).get("status") or "unknown"),
                "- Suggested phrases: " + (", ".join(phrases) if phrases else "None"),
                "",
                "### Reviewer Fields",
                "",
                "- relevant_evidence_count:",
                "- substantive_claim_count:",
                "- unsupported_claim_count:",
                "- brief_review_minutes:",
                "- answer_usefulness:",
                "- final_review_status:",
                "- notes:",
                "",
            ]
        )
    return "\n".join(sections)


def render_citations(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "- None"
    return "\n".join(
        f"- `{citation.get('citation_id')}` {citation.get('title', 'Untitled')} ({citation.get('year', 'n.d.')})"
        for citation in citations[:10]
    )


def compact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for record in records:
        payload = answer(record)
        compact.append(
            {
                "id": record.get("id"),
                "query": payload.get("query") or record.get("query"),
                "review": review(record),
                "query_diagnostics_status": (payload.get("query_diagnostics") or {}).get("stability", {}),
            }
        )
    return compact


def render_markdown(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- {'PASS' if item['passed'] else 'FAIL'}: `{item['id']}` observed={item['observed']}"
        for item in report["checks"]
    )
    blockers = "\n".join(f"- `{item}`" for item in report["release_blockers"]) or "- None"
    artifacts = "\n".join(f"- `{item}`" for item in report["required_artifacts"])
    return f"""# CANON Industry Pilot Acceptance

Status: **{report['status']}**

Mode: `{report['mode']}`

Questions prepared: {report['question_count']}

Questions reviewed: {report['reviewed_question_count']} / {report['minimum_question_count']} required

## Checks

{checks}

## Release Blockers

{blockers}

## Required Artifacts

{artifacts}
"""


def render_review_status(report: dict[str, Any]) -> str:
    incomplete = [row for row in report["rows"] if not row["complete"]]
    preview = "\n".join(
        f"- `{row['id']}` missing: {', '.join(row['missing_fields'])}"
        for row in incomplete[:30]
    ) or "- None"
    return f"""# CANON Human Review Status

Status: **{report['status']}**

Mode: `{report['mode']}`

Questions prepared: {report['question_count']}

Questions reviewed: {report['reviewed_question_count']} / {report['minimum_question_count']} required

Missing review fields: {report['missing_field_count']}

Validation errors: {report['validation_error_count']}

## Incomplete Records

{preview}
"""


def write_json(path: Path, payload: object) -> None:
    report_io.write_json(path, payload)


def write_markdown(path: Path, text: str) -> None:
    report_io.write_markdown(path, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CANON industry pilot acceptance report.")
    parser.add_argument("--mode", default=service.DEFAULT_MODE)
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument(
        "--prepare-review",
        action="store_true",
        help="Generate the 30-question human review task packet and companion acceptance reports.",
    )
    parser.add_argument(
        "--reset-review-labels",
        action="store_true",
        help="When used with --prepare-review, replace existing review labels with blank templates.",
    )
    parser.add_argument(
        "--review-status",
        action="store_true",
        help="Report which human review fields are still missing from the review task packet.",
    )
    parser.add_argument(
        "--export-review-csv",
        action="store_true",
        help="Export review fields from --records to a spreadsheet-friendly CSV.",
    )
    parser.add_argument(
        "--import-review-csv",
        type=Path,
        default=None,
        help="Import completed review fields from CSV into --records or --output.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    if args.prepare_review:
        packet = build_human_review_tasks(
            args.mode,
            preserve_existing_reviews=not args.reset_review_labels,
        )
        print(
            json.dumps(
                {
                    "report_id": packet["report_id"],
                    "mode": packet["mode"],
                    "question_count": len(packet["records"]),
                    "status": "review_tasks_written",
                    "outputs": [
                        "reports/human_review_tasks_v1.json",
                        "reports/query_diagnostics_acceptance_v1.json",
                        "reports/evidence_briefing_acceptance_v1.md",
                    ],
                },
                indent=2,
            )
        )
        return
    if args.review_status:
        report = build_review_status_report(args.mode, args.records)
        print(json.dumps(report, indent=2))
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
        report = import_review_csv(args.records, args.import_review_csv, args.output)
        print(json.dumps(report, indent=2))
        return
    report = build_industry_pilot_report(args.mode, args.records)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
