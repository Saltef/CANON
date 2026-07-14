from __future__ import annotations

from collections import Counter
from typing import Any

from canon.ingest.chunker import rough_tokens
from canon.safety.sanitizer import sanitize_retrieved_text
from canon.safety.scanner import scan_text


def expand_evidence_with_parent_context(
    evidence: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    max_parent_tokens: int = 220,
) -> list[dict[str, Any]]:
    chunks_by_id = {chunk.get("id"): chunk for chunk in chunks}
    parent_groups = chunks_by_parent_context(chunks)
    return [
        attach_parent_context(item, chunks_by_id, parent_groups, max_parent_tokens=max_parent_tokens)
        for item in evidence
    ]


def attach_parent_context(
    item: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
    parent_groups: dict[str, list[dict[str, Any]]],
    max_parent_tokens: int = 220,
) -> dict[str, Any]:
    chunk = chunks_by_id.get(item.get("chunk_id"))
    if not chunk:
        return {**item, "parent_context": blocked_parent_context("chunk_not_found")}
    importance = chunk.get("importance") or {}
    decision = parent_context_decision(item, importance)
    if not decision["allowed"]:
        return {**item, "parent_context": blocked_parent_context(decision["reason"], importance)}
    parent_id = str(importance.get("parent_context_id") or "")
    siblings = parent_groups.get(parent_id) or [chunk]
    text = parent_context_text(siblings, max_parent_tokens=max_parent_tokens)
    provenance = item_provenance(item)
    safety = scan_text(text, provenance=provenance)
    if safety.decision in {"block", "quarantine"}:
        return {
            **item,
            "parent_context": {
                **blocked_parent_context(f"parent_safety_{safety.decision}", importance),
                "safety": safety.as_dict(),
            },
        }
    if safety.decision == "sanitize":
        sanitization = sanitize_retrieved_text(text, provenance=provenance)
        text = sanitization.sanitized_text
        sanitization_payload = sanitization.to_dict()
    else:
        sanitization_payload = None
    return {
        **item,
        "parent_context": {
            "available": True,
            "included": True,
            "reason": "safe_parent_context_expansion",
            "parent_context_id": parent_id,
            "parent_context_mode": importance.get("parent_context_mode"),
            "context_expansion_policy": importance.get("context_expansion_policy"),
            "source_chunk_count": len(siblings),
            "token_count": len(rough_tokens(text)),
            "text": text,
            "safety": safety.as_dict(),
            "sanitization": sanitization_payload,
        },
    }


def parent_context_decision(item: dict[str, Any], importance: dict[str, Any]) -> dict[str, Any]:
    decision = item.get("decision") or {}
    if decision.get("safety_decision") in {"block", "quarantine"}:
        return {"allowed": False, "reason": f"chunk_safety_{decision.get('safety_decision')}"}
    if float(importance.get("parent_context_recommended", 0.0) or 0.0) < 1.0:
        return {"allowed": False, "reason": "parent_context_not_recommended"}
    if str(importance.get("context_expansion_policy") or "") != "safe_parent_expansion_allowed":
        return {"allowed": False, "reason": str(importance.get("context_expansion_policy") or "parent_context_policy_block")}
    return {"allowed": True, "reason": "safe_parent_context_expansion"}


def blocked_parent_context(reason: str, importance: dict[str, Any] | None = None) -> dict[str, Any]:
    importance = importance or {}
    return {
        "available": bool(importance.get("parent_context_available", False)),
        "included": False,
        "reason": reason,
        "parent_context_id": importance.get("parent_context_id"),
        "parent_context_mode": importance.get("parent_context_mode"),
        "context_expansion_policy": importance.get("context_expansion_policy"),
    }


def chunks_by_parent_context(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        importance = chunk.get("importance") or {}
        parent_id = str(importance.get("parent_context_id") or "")
        if not parent_id:
            continue
        groups.setdefault(parent_id, []).append(chunk)
    for parent_id, rows in groups.items():
        groups[parent_id] = sorted(rows, key=lambda chunk: (int(chunk.get("token_start") or 0), str(chunk.get("id") or "")))
    return groups


def parent_context_text(chunks: list[dict[str, Any]], max_parent_tokens: int) -> str:
    tokens: list[str] = []
    previous_end = 0
    for chunk in sorted(chunks, key=lambda row: (int(row.get("token_start") or 0), str(row.get("id") or ""))):
        chunk_tokens = rough_tokens(str(chunk.get("text") or ""))
        start = int(chunk.get("token_start") or 0)
        end = int(chunk.get("token_end") or start + len(chunk_tokens))
        skip = max(0, previous_end - start)
        tokens.extend(chunk_tokens[skip:])
        previous_end = max(previous_end, end)
        if len(tokens) >= max_parent_tokens:
            break
    return " ".join(tokens[:max_parent_tokens])


def parent_context_profile(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    groups = chunks_by_parent_context(chunks)
    eligible = 0
    roles = Counter()
    policies = Counter()
    for chunk in chunks:
        importance = chunk.get("importance") or {}
        if float(importance.get("parent_context_recommended", 0.0) or 0.0) >= 1.0:
            eligible += 1
        roles[str(importance.get("generation_context_role") or "unknown")] += 1
        policies[str(importance.get("context_expansion_policy") or "unknown")] += 1
    return {
        "parent_context_count": len(groups),
        "multi_chunk_parent_context_count": sum(1 for rows in groups.values() if len(rows) > 1),
        "parent_context_expansion_eligible_share": round(eligible / len(chunks), 6) if chunks else 0.0,
        "generation_context_role_counts": dict(sorted(roles.items())),
        "context_expansion_policy_counts": dict(sorted(policies.items())),
    }


def item_provenance(item: dict[str, Any]) -> str:
    signals = (((item.get("decision") or {}).get("stage") or {}).get("signals") or {})
    provenance = (signals.get("provenance") or {}).get("provenance")
    return str(item.get("provenance") or provenance or "academic")
