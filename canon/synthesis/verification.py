from __future__ import annotations

import re
from typing import Any

from canon.retrieval.tokenize import tokenize


CITATION_PATTERN = re.compile(r"\bC\d+\b")


def verify_generation(
    answer: str,
    citations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    context_packet: dict[str, Any],
) -> dict[str, Any]:
    valid_citation_ids = {citation["citation_id"] for citation in citations}
    evidence_by_id = {item["citation_id"]: item for item in evidence}
    allowed_ids = {
        item["citation_id"]
        for item in context_packet.get("items", [])
        if item.get("allowed_use") and item.get("evidence")
    }
    sentence_reports = [
        verify_sentence(sentence, valid_citation_ids, evidence_by_id, allowed_ids)
        for sentence in split_sentences(answer)
    ]
    if not sentence_reports:
        sentence_reports = [verify_sentence(answer, valid_citation_ids, evidence_by_id, allowed_ids)]
    invalid_citations = sorted(
        {
            citation_id
            for report in sentence_reports
            for citation_id in report["cited_ids"]
            if citation_id not in valid_citation_ids
        }
    )
    disallowed_citations = sorted(
        {
            citation_id
            for report in sentence_reports
            for citation_id in report["valid_cited_ids"]
            if citation_id not in allowed_ids
        }
    )
    unsupported_sentences = [
        report for report in sentence_reports if report["requires_support"] and not report["claim_backed"]
    ]
    blocked_leak_ids = sorted(
        {
            item.get("citation_id")
            for item in evidence
            if (item.get("decision") or {}).get("safety_decision") in {"block", "quarantine"}
            and item.get("citation_id") in answer
        }
    )
    status = "pass"
    failures = []
    if invalid_citations:
        status = "fail"
        failures.append("invalid_citations")
    if disallowed_citations:
        status = "fail"
        failures.append("disallowed_citations")
    if unsupported_sentences:
        status = "fail"
        failures.append("unsupported_sentences")
    if blocked_leak_ids:
        status = "fail"
        failures.append("blocked_or_quarantined_evidence_leaked")
    return {
        "status": status,
        "failures": failures,
        "citation_validity": 1.0 if not invalid_citations else 0.0,
        "invalid_citations": invalid_citations,
        "disallowed_citations": disallowed_citations,
        "blocked_leak_ids": blocked_leak_ids,
        "unsupported_sentence_count": len(unsupported_sentences),
        "sentence_reports": sentence_reports,
    }


def verify_sentence(
    sentence: str,
    valid_citation_ids: set[str],
    evidence_by_id: dict[str, dict[str, Any]],
    allowed_ids: set[str],
) -> dict[str, Any]:
    cited_ids = CITATION_PATTERN.findall(sentence)
    valid_ids = [citation_id for citation_id in cited_ids if citation_id in valid_citation_ids]
    sentence_tokens = content_tokens(sentence)
    evidence_tokens = set()
    attached_claim = False
    for citation_id in valid_ids:
        if citation_id not in allowed_ids:
            continue
        item = evidence_by_id.get(citation_id) or {}
        evidence_tokens.update(content_tokens(item.get("title") or ""))
        evidence_tokens.update(content_tokens(item.get("preview") or ""))
        claim = item.get("claim") or {}
        if claim:
            attached_claim = True
            evidence_tokens.update(content_tokens(claim.get("text") or ""))
    token_support = len(sentence_tokens & evidence_tokens) / len(sentence_tokens) if sentence_tokens else 1.0
    lower = sentence.lower()
    requires_support = (
        bool(sentence_tokens)
        and not lower.startswith("the current corpus does not")
        and not lower.startswith("the retrieved corpus provides weak support")
    )
    claim_backed = bool(valid_ids) and (token_support >= 0.12 or attached_claim)
    return {
        "sentence": sentence,
        "cited_ids": cited_ids,
        "valid_cited_ids": valid_ids,
        "requires_support": requires_support,
        "token_support": round(token_support, 6),
        "claim_backed": claim_backed,
    }


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def content_tokens(text: str) -> set[str]:
    stop = {
        "across",
        "answer",
        "claim",
        "cited",
        "evidence",
        "grounded",
        "retrieved",
        "reports",
        "should",
        "source",
        "that",
        "this",
        "treated",
        "with",
    }
    return {token for token in tokenize(text) if len(token) >= 4 and token not in stop}
