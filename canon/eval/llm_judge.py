from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from canon.generation.providers import get_generation_provider
from canon.product.industry_pilot import load_records
from canon.retrieval.tokenize import tokenize


QRELS_JUDGE_FIELDS = [
    "judge_relevance",
    "judge_confidence",
    "judge_rationale",
    "judge_source",
    "human_audit_priority",
]
ANSWER_JUDGE_FIELDS = [
    "id",
    "category",
    "query",
    "judge_evidence_relevance",
    "judge_citation_support",
    "judge_answer_usefulness",
    "judge_unsupported_claim_count",
    "judge_final_review_status",
    "judge_query_lingo_ratings",
    "judge_drift_risk_review",
    "judge_confidence",
    "judge_rationale",
    "judge_source",
    "human_audit_priority",
]


def judge_qrels_csv(
    csv_path: Path,
    output_path: Path,
    provider_name: str = "heuristic",
    model: str | None = None,
    fill_relevance: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = read_csv(csv_path)
    provider = None if provider_name == "heuristic" else get_generation_provider(provider_name, model)
    judged = []
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            judged.append(row)
            continue
        judgment = (
            heuristic_qrels_judgment(row)
            if provider is None
            else llm_qrels_judgment(provider, row)
        )
        updated = {**row, **judgment}
        if fill_relevance and not str(updated.get("relevance") or "").strip():
            updated["relevance"] = str(judgment["judge_relevance"])
        judged.append(updated)
    write_csv(output_path, judged, extra_fields=QRELS_JUDGE_FIELDS)
    return {
        "report_id": "llm_qrels_judge_v1",
        "status": "judged",
        "provider": provider_name,
        "model": model,
        "input_path": str(csv_path),
        "output_path": str(output_path),
        "row_count": len(rows),
        "judged_count": min(len(rows), limit) if limit is not None else len(rows),
        "fill_relevance": fill_relevance,
        "audit_summary": audit_summary(judged),
    }


def judge_answer_review_packet(
    records_path: Path,
    output_path: Path,
    provider_name: str = "heuristic",
    model: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    records = load_records(records_path)
    provider = None if provider_name == "heuristic" else get_generation_provider(provider_name, model)
    rows = []
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            continue
        judgment = (
            heuristic_answer_judgment(record)
            if provider is None
            else llm_answer_judgment(provider, record)
        )
        rows.append(
            {
                "id": record.get("id"),
                "category": record.get("category"),
                "query": record.get("query"),
                **judgment,
            }
        )
    write_csv(output_path, rows, fieldnames=ANSWER_JUDGE_FIELDS)
    return {
        "report_id": "llm_answer_review_judge_v1",
        "status": "judged",
        "provider": provider_name,
        "model": model,
        "input_path": str(records_path),
        "output_path": str(output_path),
        "record_count": len(records),
        "judged_count": len(rows),
        "audit_summary": audit_summary(rows),
        "limitations": [
            "Judge labels are provisional and must be audited by a human reviewer.",
            "Use judge output to prioritize review, not to make final production claims.",
        ],
    }


def heuristic_qrels_judgment(row: dict[str, Any]) -> dict[str, Any]:
    query_terms = content_terms(str(row.get("query") or ""))
    text_terms = content_terms(" ".join([str(row.get("title") or ""), str(row.get("preview") or "")]))
    overlap = safe_ratio(len(query_terms & text_terms), len(query_terms))
    if overlap >= 0.6:
        relevance = 3
        confidence = 0.72
        rationale = "Strong lexical overlap with query terms in title or preview."
    elif overlap >= 0.35:
        relevance = 2
        confidence = 0.62
        rationale = "Partial overlap; likely useful but needs human inspection."
    elif overlap >= 0.15:
        relevance = 1
        confidence = 0.55
        rationale = "Weak topical overlap; likely background or adjacent evidence."
    else:
        relevance = 0
        confidence = 0.64
        rationale = "Little query-term overlap; likely tangential or wrong subdomain."
    return normalize_qrels_judgment(
        {
            "judge_relevance": relevance,
            "judge_confidence": confidence,
            "judge_rationale": rationale,
            "judge_source": "heuristic_judge",
        },
        row,
    )


def heuristic_answer_judgment(record: dict[str, Any]) -> dict[str, Any]:
    answer_payload = record.get("answer_response") or {}
    evidence = answer_payload.get("evidence") or []
    citations = answer_payload.get("citations") or []
    support = answer_payload.get("support_assessment") or {}
    diagnostics = answer_payload.get("query_diagnostics") or {}
    support_level = str(support.get("support_level") or "").lower()
    evidence_count = len(evidence)
    citation_count = len(citations)
    if evidence_count >= 5 and citation_count >= 3 and support_level in {"strong", "mixed"}:
        usefulness = "usable"
        citation_support = "supported"
        relevance = "high"
        final_status = "accepted"
        confidence = 0.68
    elif evidence_count >= 3:
        usefulness = "partially_useful"
        citation_support = "partially_supported"
        relevance = "medium"
        final_status = "revised"
        confidence = 0.6
    else:
        usefulness = "not_useful"
        citation_support = "unsupported"
        relevance = "low"
        final_status = "needs_more_evidence"
        confidence = 0.62
    weak_terms = ((diagnostics.get("query_to_corpus") or {}).get("weak_terms") or [])
    drift = (diagnostics.get("stability") or {}).get("status")
    return normalize_answer_judgment(
        {
            "judge_evidence_relevance": relevance,
            "judge_citation_support": citation_support,
            "judge_answer_usefulness": usefulness,
            "judge_unsupported_claim_count": 0 if citation_support != "unsupported" else 1,
            "judge_final_review_status": final_status,
            "judge_query_lingo_ratings": "useful" if not weak_terms else "partially_useful",
            "judge_drift_risk_review": "needs_caution" if drift in {"sensitive", "unstable"} else "acceptable",
            "judge_confidence": confidence,
            "judge_rationale": "Heuristic first pass from citation count, evidence count, support level, and query diagnostics.",
            "judge_source": "heuristic_judge",
        },
        record,
    )


def llm_qrels_judgment(provider: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""You are judging retrieval relevance for a research evidence system.
Return only JSON with keys: judge_relevance, judge_confidence, judge_rationale.

Scale:
3 = directly answers or strongly supports the query
2 = partially useful evidence or important qualification
1 = background/context only
0 = not relevant, misleading, or too tangential

Query: {row.get('query')}
Candidate title: {row.get('title')}
Source: {row.get('source_name')} ({row.get('year')})
Candidate preview: {row.get('preview')}
"""
    generated = provider.generate(prompt)
    judgment = parse_json_object(generated.text)
    judgment["judge_source"] = f"llm_judge:{generated.provider}:{generated.model}"
    return normalize_qrels_judgment(judgment, row)


def llm_answer_judgment(provider: Any, record: dict[str, Any]) -> dict[str, Any]:
    answer_payload = record.get("answer_response") or {}
    prompt = f"""You are judging a cited answer from a research discovery system.
Return only JSON with keys:
judge_evidence_relevance, judge_citation_support, judge_answer_usefulness,
judge_unsupported_claim_count, judge_final_review_status,
judge_query_lingo_ratings, judge_drift_risk_review, judge_confidence, judge_rationale.

Allowed labels:
judge_evidence_relevance: high, medium, low
judge_citation_support: supported, partially_supported, unsupported
judge_answer_usefulness: usable, partially_useful, not_useful
judge_final_review_status: accepted, revised, rejected, needs_more_evidence
judge_query_lingo_ratings: useful, partially_useful, not_useful
judge_drift_risk_review: acceptable, needs_caution, drifted

Query: {record.get('query')}
Answer: {answer_payload.get('answer')}
Citations: {json.dumps(answer_payload.get('citations') or [], ensure_ascii=False)[:4000]}
Evidence previews: {json.dumps(answer_payload.get('evidence') or [], ensure_ascii=False)[:6000]}
Support assessment: {json.dumps(answer_payload.get('support_assessment') or {}, ensure_ascii=False)}
Query diagnostics: {json.dumps(answer_payload.get('query_diagnostics') or {}, ensure_ascii=False)[:4000]}
"""
    generated = provider.generate(prompt)
    judgment = parse_json_object(generated.text)
    judgment["judge_source"] = f"llm_judge:{generated.provider}:{generated.model}"
    return normalize_answer_judgment(judgment, record)


def normalize_qrels_judgment(judgment: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    relevance = int(float(judgment.get("judge_relevance", 0)))
    relevance = min(3, max(0, relevance))
    confidence = min(1.0, max(0.0, float(judgment.get("judge_confidence", 0.0) or 0.0)))
    audit_priority = qrels_audit_priority(row, relevance, confidence)
    return {
        "judge_relevance": str(relevance),
        "judge_confidence": round(confidence, 3),
        "judge_rationale": str(judgment.get("judge_rationale") or "")[:600],
        "judge_source": str(judgment.get("judge_source") or "llm_judge"),
        "human_audit_priority": audit_priority,
    }


def normalize_answer_judgment(judgment: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    confidence = min(1.0, max(0.0, float(judgment.get("judge_confidence", 0.0) or 0.0)))
    unsupported = int(float(judgment.get("judge_unsupported_claim_count", 0) or 0))
    audit_priority = answer_audit_priority(record, judgment, confidence)
    return {
        "judge_evidence_relevance": label(judgment, "judge_evidence_relevance", {"high", "medium", "low"}, "medium"),
        "judge_citation_support": label(
            judgment,
            "judge_citation_support",
            {"supported", "partially_supported", "unsupported"},
            "partially_supported",
        ),
        "judge_answer_usefulness": label(
            judgment,
            "judge_answer_usefulness",
            {"usable", "partially_useful", "not_useful"},
            "partially_useful",
        ),
        "judge_unsupported_claim_count": max(0, unsupported),
        "judge_final_review_status": label(
            judgment,
            "judge_final_review_status",
            {"accepted", "revised", "rejected", "needs_more_evidence"},
            "revised",
        ),
        "judge_query_lingo_ratings": label(
            judgment,
            "judge_query_lingo_ratings",
            {"useful", "partially_useful", "not_useful"},
            "partially_useful",
        ),
        "judge_drift_risk_review": label(
            judgment,
            "judge_drift_risk_review",
            {"acceptable", "needs_caution", "drifted"},
            "needs_caution",
        ),
        "judge_confidence": round(confidence, 3),
        "judge_rationale": str(judgment.get("judge_rationale") or "")[:600],
        "judge_source": str(judgment.get("judge_source") or "llm_judge"),
        "human_audit_priority": audit_priority,
    }


def qrels_audit_priority(row: dict[str, Any], relevance: int, confidence: float) -> str:
    best_rank = int(float(row.get("best_rank") or 9999))
    if confidence < 0.6:
        return "high"
    if best_rank <= 3 and relevance == 0:
        return "high"
    if relevance == 3:
        return "medium"
    return "low"


def answer_audit_priority(record: dict[str, Any], judgment: dict[str, Any], confidence: float) -> str:
    category = str(record.get("category") or "")
    if confidence < 0.6:
        return "high"
    if category in {"overclaim_check", "weak_support", "mixed_evidence"}:
        return "high"
    if judgment.get("judge_unsupported_claim_count", 0):
        return "high"
    if judgment.get("judge_final_review_status") == "accepted":
        return "medium"
    return "low"


def label(judgment: dict[str, Any], key: str, allowed: set[str], default: str) -> str:
    value = str(judgment.get(key) or "").strip()
    return value if value in allowed else default


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Judge response did not contain a JSON object.")
    return json.loads(stripped[start : end + 1])


def content_terms(text: str) -> set[str]:
    stop = {
        "about",
        "after",
        "and",
        "are",
        "can",
        "does",
        "for",
        "from",
        "how",
        "into",
        "the",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    return {token for token in tokenize(text) if len(token) >= 3 and token not in stop}


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def audit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        priority = str(row.get("human_audit_priority") or "unknown")
        counts[priority] = counts.get(priority, 0) + 1
    return {"human_audit_priority_counts": dict(sorted(counts.items()))}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    extra_fields: list[str] | None = None,
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        base_fields = list(rows[0].keys()) if rows else []
        for field in extra_fields or []:
            if field not in base_fields:
                base_fields.append(field)
        fieldnames = base_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run provisional LLM-assisted judges for CANON review tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    qrels = subparsers.add_parser("qrels")
    qrels.add_argument("--csv", required=True)
    qrels.add_argument("--output", required=True)
    qrels.add_argument("--provider", default="heuristic")
    qrels.add_argument("--model", default=None)
    qrels.add_argument("--fill-relevance", action="store_true")
    qrels.add_argument("--limit", type=int, default=None)
    answers = subparsers.add_parser("answers")
    answers.add_argument("--records", required=True)
    answers.add_argument("--output", required=True)
    answers.add_argument("--provider", default="heuristic")
    answers.add_argument("--model", default=None)
    answers.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.command == "qrels":
        report = judge_qrels_csv(
            csv_path=Path(args.csv),
            output_path=Path(args.output),
            provider_name=args.provider,
            model=args.model,
            fill_relevance=args.fill_relevance,
            limit=args.limit,
        )
    else:
        report = judge_answer_review_packet(
            records_path=Path(args.records),
            output_path=Path(args.output),
            provider_name=args.provider,
            model=args.model,
            limit=args.limit,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
