from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from canon.retrieval.tokenize import tokenize
from canon.safety.sanitizer import sanitize_retrieved_text


CONTEXT_ROLE_ORDER = [
    "direct_evidence",
    "method_detail",
    "limitation_or_caveat",
    "corroborating_source",
    "contradicting_source",
    "contextual_background",
]


@dataclass(frozen=True)
class ContextItem:
    citation_id: str
    role: str
    allowed_use: str
    evidence: dict[str, Any]
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def assemble_context_packet(
    query: str,
    evidence: list[dict[str, Any]],
    corroboration: dict | None = None,
    conflicts: list[dict] | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    policies = {item.get("citation_id"): evidence_use_policy(item) for item in evidence}
    safe_evidence = [
        item
        for item in evidence
        if policies.get(item.get("citation_id"), {}).get("generator_context_allowed")
    ]
    role_candidates = [(context_role(query, item, corroboration, conflicts), item) for item in safe_evidence]
    selected: list[ContextItem] = []
    used_ids: set[str] = set()
    limit = max_items or len(safe_evidence)

    for role in CONTEXT_ROLE_ORDER:
        if len(selected) >= limit:
            break
        candidates = [item for item_role, item in role_candidates if item_role == role]
        if not candidates:
            continue
        best = sorted(candidates, key=context_sort_key, reverse=True)[0]
        if best["citation_id"] not in used_ids:
            selected.append(context_item(query, best, role))
            used_ids.add(best["citation_id"])

    for _role, item in sorted(role_candidates, key=lambda row: context_sort_key(row[1]), reverse=True):
        if len(selected) >= limit:
            break
        if item["citation_id"] in used_ids:
            continue
        selected.append(context_item(query, item, context_role(query, item, corroboration, conflicts)))
        used_ids.add(item["citation_id"])

    excluded = [
        {
            "citation_id": item.get("citation_id"),
            "chunk_id": item.get("chunk_id"),
            "reason": policies.get(item.get("citation_id"), {}).get("reason", "not_allowed_for_context"),
            "allowed_uses": policies.get(item.get("citation_id"), {}).get("allowed_uses", ["audit"]),
            "decision": item.get("decision", {}),
        }
        for item in evidence
        if item not in safe_evidence
    ]
    return {
        "query": query,
        "items": [item.to_dict() for item in selected],
        "excluded": excluded,
        "assembly_summary": {
            "input_evidence_count": len(evidence),
            "safe_evidence_count": len(safe_evidence),
            "generator_context_eligible_count": len(safe_evidence),
            "aggregate_only_count": sum(1 for policy in policies.values() if "aggregate_public_opinion" in policy.get("allowed_uses", [])),
            "selected_count": len(selected),
            "role_counts": role_counts(selected),
            "excluded_count": len(excluded),
        },
    }


def evidence_allowed_for_context(item: dict[str, Any]) -> bool:
    return bool(evidence_use_policy(item)["generator_context_allowed"])


def evidence_use_policy(item: dict[str, Any]) -> dict[str, Any]:
    decision = item.get("decision") or {}
    explicit_allowed_uses = list(item.get("allowed_uses") or [])
    if explicit_allowed_uses:
        return {
            "generator_context_allowed": "generator_context" in explicit_allowed_uses,
            "allowed_uses": explicit_allowed_uses,
            "reason": "explicit_allowed_uses",
        }
    safety_decision = decision.get("safety_decision", "allow")
    if safety_decision in {"block", "quarantine"}:
        return {
            "generator_context_allowed": False,
            "allowed_uses": ["audit"],
            "reason": f"safety_{safety_decision}",
        }
    if decision.get("decision") == "reject":
        return {
            "generator_context_allowed": False,
            "allowed_uses": ["audit"],
            "reason": "retrieval_decision_reject",
        }
    signals = ((decision.get("stage") or {}).get("signals") or {})
    document_profile = signals.get("document_profile") or {}
    domain_profile = signals.get("domain_profile") or {}
    evidence_family = str(document_profile.get("evidence_family") or "")
    recommended_role = str(document_profile.get("recommended_decision_role") or "")
    gate_flags = set(domain_profile.get("gate_flags") or [])
    domain_tier = str(domain_profile.get("domain_fit_tier") or "")
    if evidence_family == "public_opinion" or "aggregation_required" in gate_flags or recommended_role == "aggregate_public_opinion_and_screen_safety":
        return {
            "generator_context_allowed": False,
            "allowed_uses": ["audit", "aggregate_public_opinion"],
            "reason": "public_opinion_aggregate_only",
        }
    if domain_tier in {"blocked", "quarantine_review"}:
        return {
            "generator_context_allowed": False,
            "allowed_uses": ["audit"],
            "reason": f"domain_fit_{domain_tier}",
        }
    uses = ["audit", "generator_context"]
    if safety_decision == "sanitize":
        uses.append("sanitized_context")
    return {
        "generator_context_allowed": True,
        "allowed_uses": uses,
        "reason": "generator_context_allowed",
    }


def context_role(
    query: str,
    item: dict[str, Any],
    corroboration: dict | None,
    conflicts: list[dict] | None,
) -> str:
    components = item.get("components") or {}
    claim = item.get("claim") or {}
    if item.get("conflict_injected") or claim.get("stance") == "critical" or has_conflict_link(item, conflicts):
        return "contradicting_source"
    if float(components.get("limitation_signal", 0.0)) >= 0.35:
        return "limitation_or_caveat"
    if float(components.get("method_signal", 0.0)) >= 0.35:
        return "method_detail"
    if is_corroborating(item, corroboration):
        return "corroborating_source"
    if query_overlap(query, item) >= 0.25 or float(components.get("relevance", 0.0)) > 0.0:
        return "direct_evidence"
    return "contextual_background"


def context_item(query: str, item: dict[str, Any], role: str) -> ContextItem:
    evidence = evidence_for_context(item)
    return ContextItem(
        citation_id=item["citation_id"],
        role=role,
        allowed_use=allowed_use_for_role(role),
        evidence=evidence,
        reasons=context_reasons(query, item, role),
    )


def evidence_for_context(item: dict[str, Any]) -> dict[str, Any]:
    decision = item.get("decision") or {}
    if decision.get("safety_decision") != "sanitize":
        return item
    sanitized = dict(item)
    provenance = str(item.get("provenance") or (decision.get("stage") or {}).get("signals", {}).get("provenance", {}).get("provenance") or "academic")
    preview_sanitization = sanitize_retrieved_text(item.get("preview") or "", provenance=provenance)
    sanitized["raw_preview"] = item.get("preview")
    sanitized["preview"] = preview_sanitization.sanitized_text
    sanitized["sanitization"] = preview_sanitization.to_dict()
    claim = item.get("claim")
    if isinstance(claim, dict) and claim.get("text"):
        claim_sanitization = sanitize_retrieved_text(str(claim["text"]), provenance=provenance)
        sanitized["claim"] = {**claim, "raw_text": claim.get("text"), "text": claim_sanitization.sanitized_text}
        sanitized["claim_sanitization"] = claim_sanitization.to_dict()
    return sanitized


def allowed_use_for_role(role: str) -> str:
    if role == "contradicting_source":
        return "use_for_conflict_or_caveat"
    if role == "limitation_or_caveat":
        return "use_for_limitations"
    if role == "method_detail":
        return "use_for_method_context"
    return "use_as_supporting_evidence"


def context_reasons(query: str, item: dict[str, Any], role: str) -> list[str]:
    reasons = [role]
    overlap = query_overlap(query, item)
    if overlap > 0:
        reasons.append(f"query_overlap={overlap:.3f}")
    components = item.get("components") or {}
    for key in ["relevance", "semantic_similarity", "method_signal", "limitation_signal", "evidence_specificity"]:
        value = float(components.get(key, 0.0))
        if value > 0:
            reasons.append(f"{key}={value:.3f}")
    return reasons


def context_sort_key(item: dict[str, Any]) -> tuple[float, float, float, int]:
    components = item.get("components") or {}
    score = float(item.get("final_score") or 0.0)
    relevance = max(float(components.get("relevance", 0.0)), float(components.get("semantic_similarity", 0.0)))
    specificity = float(components.get("evidence_specificity", 0.0))
    return (score, relevance, specificity, -int(item.get("rank") or 9999))


def query_overlap(query: str, item: dict[str, Any]) -> float:
    query_terms = {token for token in tokenize(query) if len(token) >= 4}
    if not query_terms:
        return 0.0
    text = " ".join([item.get("title") or "", item.get("preview") or "", (item.get("claim") or {}).get("text") or ""])
    text_terms = set(tokenize(text))
    return len(query_terms & text_terms) / len(query_terms)


def is_corroborating(item: dict[str, Any], corroboration: dict | None) -> bool:
    if not corroboration:
        return False
    work_id = str(item.get("work_id") or "")
    for group in corroboration.get("groups") or []:
        if group.get("verdict") == "independently_corroborated" and work_id in set(group.get("work_ids") or []):
            return True
    return False


def has_conflict_link(item: dict[str, Any], conflicts: list[dict] | None) -> bool:
    if not conflicts:
        return False
    claim = item.get("claim") or {}
    claim_id = claim.get("id")
    if not claim_id:
        return False
    return any(claim_id in {conflict.get("left_claim_id"), conflict.get("right_claim_id")} for conflict in conflicts)


def role_counts(items: list[ContextItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.role] = counts.get(item.role, 0) + 1
    return dict(sorted(counts.items()))
