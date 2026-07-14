from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from canon.retrieval.tokenize import tokenize


DEPENDENCE_MARKERS = {
    "press release",
    "reprinted",
    "syndicated",
    "copied",
    "mirror",
    "same dataset",
    "same source",
    "according to",
}
SUPPORTIVE_STANCES = {"supportive", "descriptive", "mixed"}
CONTRADICTORY_STANCES = {"critical"}


@dataclass(frozen=True)
class CorroborationGroup:
    claim_key: str
    stance_counts: dict[str, int]
    support_count: int
    independent_support_count: int
    contradiction_count: int
    dependent_echo_count: int
    source_fingerprints: list[str]
    work_ids: list[str]
    echo_risk: float
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


def assess_corroboration(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    groups = []
    by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        claim_key = claim_group_key(item)
        if claim_key:
            by_claim[claim_key].append(item)
    for claim_key, items in sorted(by_claim.items()):
        groups.append(build_group(claim_key, items))
    summary = summarize_groups(groups)
    return {
        "group_count": len(groups),
        "groups": [group.to_dict() for group in groups],
        "summary": summary,
    }


def build_group(claim_key: str, items: list[dict[str, Any]]) -> CorroborationGroup:
    stance_counts = Counter(normalized_stance(item) for item in items)
    supporting = [item for item in items if normalized_stance(item) in SUPPORTIVE_STANCES]
    contradictory = [item for item in items if normalized_stance(item) in CONTRADICTORY_STANCES]
    fingerprints = [source_fingerprint(item) for item in supporting]
    independent_fingerprints = sorted(set(fingerprints))
    dependent_echo_count = max(0, len(fingerprints) - len(independent_fingerprints))
    echo_risk = echo_risk_score(items, dependent_echo_count, independent_fingerprints)
    independent_support_count = len(independent_fingerprints)
    return CorroborationGroup(
        claim_key=claim_key,
        stance_counts=dict(sorted(stance_counts.items())),
        support_count=len(supporting),
        independent_support_count=independent_support_count,
        contradiction_count=len(contradictory),
        dependent_echo_count=dependent_echo_count,
        source_fingerprints=independent_fingerprints,
        work_ids=sorted({str(item.get("work_id") or "") for item in items if item.get("work_id")}),
        echo_risk=echo_risk,
        verdict=group_verdict(independent_support_count, len(contradictory), echo_risk),
    )


def summarize_groups(groups: list[CorroborationGroup]) -> dict[str, Any]:
    if not groups:
        return {
            "max_independent_support": 0,
            "total_support_count": 0,
            "total_independent_support": 0,
            "total_contradictions": 0,
            "max_echo_risk": 0.0,
            "verdict_counts": {},
            "overall_verdict": "no_claims",
        }
    verdict_counts = Counter(group.verdict for group in groups)
    max_echo_risk = max(group.echo_risk for group in groups)
    total_independent = sum(group.independent_support_count for group in groups)
    total_contradictions = sum(group.contradiction_count for group in groups)
    if any(group.verdict == "adversarial_echo_risk" for group in groups):
        overall = "adversarial_echo_risk"
    elif total_contradictions:
        overall = "conflicted"
    elif any(group.independent_support_count >= 2 for group in groups):
        overall = "independently_corroborated"
    elif any(group.support_count for group in groups):
        overall = "single_source_or_dependent"
    else:
        overall = "no_support"
    return {
        "max_independent_support": max(group.independent_support_count for group in groups),
        "total_support_count": sum(group.support_count for group in groups),
        "total_independent_support": total_independent,
        "total_contradictions": total_contradictions,
        "max_echo_risk": round(max_echo_risk, 6),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "overall_verdict": overall,
    }


def claim_group_key(item: dict[str, Any]) -> str:
    claim = item.get("claim") or {}
    text = claim.get("text") or item.get("preview") or item.get("title") or ""
    tokens = [
        token
        for token in tokenize(text)
        if len(token) >= 5 and token not in {"about", "across", "claim", "evidence", "finds", "shows", "study"}
    ]
    if not tokens:
        return ""
    return " ".join(sorted(set(tokens))[:8])


def normalized_stance(item: dict[str, Any]) -> str:
    stance = ((item.get("claim") or {}).get("stance") or item.get("stance") or "descriptive").lower()
    if stance not in SUPPORTIVE_STANCES | CONTRADICTORY_STANCES:
        return "descriptive"
    return stance


def source_fingerprint(item: dict[str, Any]) -> str:
    source = str(item.get("source_name") or "UNKNOWN").lower()
    title = str(item.get("title") or "").lower()
    authors = item.get("author_display_names") or item.get("authors") or []
    author_key = "|".join(sorted(str(author).lower() for author in authors[:3]))
    provenance = str(item.get("provenance") or item.get("source_type") or "").lower()
    dependence_key = explicit_dependence_key(item)
    if dependence_key:
        return dependence_key
    if author_key:
        return f"{source}:{author_key}"
    normalized_title = re.sub(r"[^a-z0-9]+", " ", title).strip()
    return f"{source}:{normalized_title[:48]}:{provenance}"


def explicit_dependence_key(item: dict[str, Any]) -> str:
    for key in ["source_chain_id", "original_source_id", "dataset_id", "press_release_id"]:
        value = item.get(key) or (item.get("metadata") or {}).get(key)
        if value:
            return f"{key}:{value}"
    return ""


def echo_risk_score(
    items: list[dict[str, Any]],
    dependent_echo_count: int,
    independent_fingerprints: list[str],
) -> float:
    if not items:
        return 0.0
    repeated_text_risk = near_duplicate_risk(items)
    dependence_risk = dependent_echo_count / max(1, len(items))
    concentration_risk = 1.0 - (len(independent_fingerprints) / max(1, len(items)))
    marker_risk = dependence_marker_risk(items)
    score = (
        (dependence_risk * 0.35)
        + (concentration_risk * 0.25)
        + (repeated_text_risk * 0.25)
        + (marker_risk * 0.15)
    )
    return round(min(1.0, score), 6)


def near_duplicate_risk(items: list[dict[str, Any]]) -> float:
    texts = [claim_or_preview(item) for item in items]
    if len(texts) < 2:
        return 0.0
    pairs = 0
    high_overlap = 0
    token_sets = [set(tokenize(text)) for text in texts]
    for left_index, left in enumerate(token_sets):
        for right in token_sets[left_index + 1 :]:
            pairs += 1
            if not left or not right:
                continue
            jaccard = len(left & right) / len(left | right)
            if jaccard >= 0.72:
                high_overlap += 1
    return high_overlap / pairs if pairs else 0.0


def dependence_marker_risk(items: list[dict[str, Any]]) -> float:
    hits = 0
    for item in items:
        text = claim_or_preview(item).lower()
        if any(marker in text for marker in DEPENDENCE_MARKERS):
            hits += 1
    return hits / len(items) if items else 0.0


def claim_or_preview(item: dict[str, Any]) -> str:
    claim = item.get("claim") or {}
    return str(claim.get("text") or item.get("preview") or item.get("title") or "")


def group_verdict(independent_support_count: int, contradiction_count: int, echo_risk: float) -> str:
    if echo_risk >= 0.45 and independent_support_count <= 1:
        return "adversarial_echo_risk"
    if contradiction_count:
        return "conflicted"
    if independent_support_count >= 2:
        return "independently_corroborated"
    if independent_support_count == 1:
        return "single_source_or_dependent"
    return "no_support"
