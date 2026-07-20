from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Any

from canon.config import load_settings
from canon.product import report_io, service
from canon.product.research_workflow import run_research_workflow
from canon.retrieval.tokenize import tokenize


def run_stage2_synthesis(payload: dict[str, Any]) -> dict[str, Any]:
    question = service.optional_text(payload, "question") or service.require_text(payload, "query")
    workflow = service.optional_dict(payload.get("research_workflow"), "research_workflow")
    if not workflow:
        workflow = run_research_workflow({**payload, "question": question, "write_report": False})
    evidence = synthesis_evidence(workflow)
    claims = build_cited_claims(question, evidence, max_claims=service.optional_int(payload.get("max_claims")) or 6)
    synthesis = build_synthesis(question, claims, evidence, workflow)
    gate = stage2_quality_gate(
        claims=claims,
        evidence=evidence,
        workflow=workflow,
        min_evidence_count=service.optional_int(payload.get("min_evidence_count")) or 2,
        min_distinct_sources=service.optional_int(payload.get("min_distinct_sources")) or 2,
    )
    report = {
        "report_id": "stage2_evidence_synthesis_v1",
        "status": stage2_status(gate, evidence),
        "question": question,
        "mode": workflow.get("mode") or str(payload.get("mode") or service.DEFAULT_MODE),
        "policy": workflow.get("policy") or str(payload.get("policy") or service.DEFAULT_POLICY),
        "stage1_report_id": workflow.get("report_id"),
        "stage1_status": workflow.get("status"),
        "synthesis": synthesis,
        "cited_claims": claims,
        "evidence": evidence,
        "quality_gate": gate,
        "human_review_required": True,
        "production_boundary": [
            "Stage 2 creates a review-ready synthesis from retrieved evidence.",
            "Automated checks verify citation integrity and evidence overlap, not final factual correctness.",
            "Human review is still required before release-quality claims.",
        ],
    }
    if service.optional_bool(payload.get("write_report"), default=True):
        write_report(report)
    return report


def synthesis_evidence(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    packet_response = workflow.get("evidence_packet_response") or {}
    packets = packet_response.get("evidence_packets") or []
    if not packets:
        return []
    rows = []
    for item in packets[0].get("supporting_evidence") or []:
        evidence_id = str(item.get("evidence_id") or item.get("citation_id") or f"C{len(rows) + 1}")
        rows.append(
            {
                "evidence_id": evidence_id,
                "chunk_id": str(item.get("chunk_id") or ""),
                "document_id": str(item.get("document_id") or item.get("work_id") or ""),
                "title": str(item.get("title") or ""),
                "source_name": str(item.get("source_name") or "unknown"),
                "source_type": str(item.get("source_type") or "unknown"),
                "language": str(item.get("language") or "unknown"),
                "domain": str(item.get("domain") or "unknown"),
                "cluster_id": str(item.get("cluster_id") or "unknown"),
                "text": evidence_text(item),
                "metadata": {
                    "evidence_scope": item.get("evidence_scope"),
                    "retrieval_stage": item.get("retrieval_stage"),
                    "rank": item.get("rank"),
                },
            }
        )
    return rows


def evidence_text(item: dict[str, Any]) -> str:
    text = item.get("text") or item.get("preview") or item.get("claim") or ""
    return compact_whitespace(str(text))[:900]


def build_cited_claims(question: str, evidence: list[dict[str, Any]], max_claims: int = 6) -> list[dict[str, Any]]:
    question_terms = content_terms(question)
    claims = []
    for item in evidence[:max_claims]:
        focus = best_evidence_sentence(item["text"], question_terms) or item["text"][:220]
        claim_text = compact_sentence(focus)
        if not claim_text:
            claim_text = f"{item['title']} provides contextual evidence."
        claims.append(
            {
                "claim_id": f"S2C{len(claims) + 1}",
                "claim": claim_text,
                "evidence_ids": [item["evidence_id"]],
                "support_status": "cited",
                "confidence": claim_confidence(claim_text, item, question_terms),
                "source_name": item["source_name"],
            }
        )
    return claims


def best_evidence_sentence(text: str, question_terms: set[str]) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    ranked = sorted(
        sentences,
        key=lambda sentence: (len(content_terms(sentence) & question_terms), len(content_terms(sentence))),
        reverse=True,
    )
    return ranked[0]


def claim_confidence(claim: str, evidence: dict[str, Any], question_terms: set[str]) -> float:
    claim_terms = content_terms(claim)
    evidence_terms = content_terms(" ".join([evidence["title"], evidence["text"]]))
    support_overlap = safe_ratio(len(claim_terms & evidence_terms), len(claim_terms))
    focus_overlap = safe_ratio(len(question_terms & evidence_terms), len(question_terms))
    return round(min(0.95, 0.35 + 0.45 * support_overlap + 0.20 * focus_overlap), 6)


def build_synthesis(
    question: str,
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    workflow: dict[str, Any],
) -> dict[str, Any]:
    answer_sentences = [
        f"{claim['claim']} [{', '.join(claim['evidence_ids'])}]"
        for claim in claims[:4]
    ]
    if not answer_sentences:
        answer_sentences.append("The current evidence packet does not contain enough evidence to synthesize an answer.")
    return {
        "answer": " ".join(answer_sentences),
        "summary_claim_count": len(claims),
        "citation_count": sum(len(claim["evidence_ids"]) for claim in claims),
        "evidence_scope": evidence_scope(evidence),
        "source_summary": source_summary(evidence),
        "contradictions_and_uncertainty": uncertainty_notes(workflow, evidence),
        "next_review_questions": next_review_questions(question, workflow, evidence),
    }


def stage2_quality_gate(
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    workflow: dict[str, Any],
    min_evidence_count: int = 2,
    min_distinct_sources: int = 2,
) -> dict[str, Any]:
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    checks = [
        check("evidence_present", len(evidence) >= min_evidence_count, {"evidence_count": len(evidence)}),
        check(
            "source_breadth",
            len({row["source_name"] for row in evidence}) >= min_distinct_sources,
            {"distinct_sources": len({row["source_name"] for row in evidence})},
        ),
        check("citation_integrity", all_known_citations(claims, evidence_by_id), {}),
        check("claim_evidence_overlap", unsupported_claim_count(claims, evidence_by_id) == 0, {
            "unsupported_claim_count": unsupported_claim_count(claims, evidence_by_id),
        }),
        check(
            "stage1_completed",
            workflow.get("status") not in {"failed", "empty"},
            {"stage1_status": workflow.get("status")},
        ),
    ]
    blockers = [row for row in checks if row["status"] == "fail" and row["id"] in {"evidence_present", "citation_integrity", "claim_evidence_overlap"}]
    return {
        "status": "pass_pending_human_review" if not blockers else "blocked",
        "checks": checks,
        "unsupported_claims": unsupported_claims(claims, evidence_by_id),
        "human_review_required": True,
    }


def all_known_citations(claims: list[dict[str, Any]], evidence_by_id: dict[str, dict[str, Any]]) -> bool:
    return all(
        evidence_id in evidence_by_id
        for claim in claims
        for evidence_id in claim.get("evidence_ids", [])
    )


def unsupported_claims(claims: list[dict[str, Any]], evidence_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for claim in claims:
        evidence_terms = set()
        for evidence_id in claim.get("evidence_ids", []):
            evidence = evidence_by_id.get(str(evidence_id), {})
            evidence_terms |= content_terms(" ".join([str(evidence.get("title") or ""), str(evidence.get("text") or "")]))
        claim_terms = content_terms(str(claim.get("claim") or ""))
        overlap = safe_ratio(len(claim_terms & evidence_terms), len(claim_terms))
        if overlap < 0.5:
            rows.append({"claim_id": claim.get("claim_id"), "overlap": round(overlap, 6)})
    return rows


def unsupported_claim_count(claims: list[dict[str, Any]], evidence_by_id: dict[str, dict[str, Any]]) -> int:
    return len(unsupported_claims(claims, evidence_by_id))


def stage2_status(gate: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "blocked_insufficient_evidence"
    if gate["status"] == "blocked":
        return "blocked_quality_gate"
    return "ready_for_human_review"


def evidence_scope(evidence: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item.get("metadata", {}).get("evidence_scope") or "unknown") for item in evidence)
    return dict(sorted(counts.items()))


def source_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "distinct_sources": len({item["source_name"] for item in evidence}),
        "distinct_source_types": len({item["source_type"] for item in evidence}),
        "distinct_domains": len({item["domain"] for item in evidence}),
        "source_counts": dict(sorted(Counter(item["source_name"] for item in evidence).items())),
    }


def uncertainty_notes(workflow: dict[str, Any], evidence: list[dict[str, Any]]) -> list[str]:
    notes = []
    packet_response = workflow.get("evidence_packet_response") or {}
    for gap in packet_response.get("coverage_gaps") or []:
        notes.append(str(gap.get("gap") or gap))
    if len(evidence) < 3:
        notes.append("The synthesis is based on fewer than three evidence items.")
    if len({item["source_name"] for item in evidence}) < 2:
        notes.append("The evidence packet has limited independent source breadth.")
    guidance = ((workflow.get("layers") or {}).get("research_guidance_layer") or {})
    gate = guidance.get("synthesis_gate") or {}
    if gate.get("status"):
        notes.append(f"Stage 1 synthesis gate: {gate.get('status')}.")
    return dedupe(notes)[:8]


def next_review_questions(question: str, workflow: dict[str, Any], evidence: list[dict[str, Any]]) -> list[str]:
    questions = []
    guidance = ((workflow.get("layers") or {}).get("research_guidance_layer") or {})
    for action in guidance.get("next_actions") or []:
        questions.append(str(action.get("description") or action.get("id") or "Review evidence gap."))
    if evidence:
        questions.append("Do the cited evidence items directly support the synthesis claims?")
    else:
        questions.append(f"What additional retrieval is needed before answering: {question}")
    return dedupe(questions)[:6]


def check(check_id: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"id": check_id, "status": "pass" if passed else "fail", "details": details}


def content_terms(text: str) -> set[str]:
    stop = {
        "about",
        "and",
        "are",
        "based",
        "does",
        "from",
        "into",
        "the",
        "this",
        "with",
    }
    return {token for token in tokenize(text) if len(token) >= 3 and token not in stop}


def split_sentences(text: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()]


def compact_sentence(text: str, limit: int = 220) -> str:
    value = compact_whitespace(text)
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0].rstrip(".,;:") + "."


def compact_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    rows = []
    for value in values:
        normalized = compact_whitespace(value).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        rows.append(value)
    return rows


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def write_report(report: dict[str, Any]) -> None:
    settings = load_settings()
    slug = "".join(character if character.isalnum() else "-" for character in report["question"].lower())[:60].strip("-")
    output = settings.reports_dir / f"stage2_synthesis_{report['mode']}_{slug or 'query'}.json"
    report_io.write_json(output, report)
    report_io.write_markdown(output.with_suffix(".md"), render_markdown(report))
    report["output_path"] = str(output).replace("\\", "/")


def render_markdown(report: dict[str, Any]) -> str:
    claims = "\n".join(
        f"- `{claim['claim_id']}` {claim['claim']} ({', '.join(claim['evidence_ids'])})"
        for claim in report["cited_claims"]
    )
    checks = "\n".join(f"- `{row['id']}`: `{row['status']}`" for row in report["quality_gate"]["checks"])
    return f"""# Stage 2 Evidence Synthesis

Status: `{report['status']}`

Question: {report['question']}

## Answer

{report['synthesis']['answer']}

## Cited Claims

{claims}

## Quality Gate

{checks}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 evidence synthesis.")
    parser.add_argument("query")
    parser.add_argument("--mode", default=service.DEFAULT_MODE)
    parser.add_argument("--policy", default=service.DEFAULT_POLICY)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage2_synthesis(
                {
                    "query": args.query,
                    "mode": args.mode,
                    "policy": args.policy,
                    "top_k": args.top_k,
                    "write_report": args.write_report,
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
