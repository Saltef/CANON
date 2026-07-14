from __future__ import annotations

import hashlib
import re
from typing import Any

from canon.models import Chunk, Work
from canon.quality.document_types import document_type_profile


TOKEN_RE = re.compile(r"\S+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


DOCUMENT_TYPE_CHUNK_POLICIES = {
    "academic_article": {
        "strategy": "section_sentence_boundary",
        "rationale": "Academic evidence often needs claim, method, result, and limitation spans preserved together.",
    },
    "preprint": {
        "strategy": "section_sentence_boundary",
        "rationale": "Preprint chunks use the academic policy while review status remains a source-trust signal.",
    },
    "policy_report": {
        "strategy": "section_sentence_boundary",
        "max_tokens": 220,
        "rationale": "Policy reports often mix findings, caveats, and recommendations in section-level prose.",
    },
    "market_report": {
        "strategy": "section_sentence_boundary",
        "max_tokens": 220,
        "rationale": "Market reports need enough local context for estimates, assumptions, and disclosed incentives.",
    },
    "news_article": {
        "strategy": "paragraph_sentence_boundary",
        "max_tokens": 180,
        "rationale": "News chunks should preserve attribution, quote context, and chronology without becoming broad summaries.",
    },
    "web_page": {
        "strategy": "web_section_boundary",
        "max_tokens": 180,
        "rationale": "Open-web chunks get smaller windows because extraction noise and adversarial surface are higher.",
    },
    "legal_authority": {
        "strategy": "legal_clause_boundary",
        "max_tokens": 180,
        "min_overlap": 24,
        "rationale": "Legal text needs clauses, exceptions, citations, dates, and jurisdictional qualifiers kept near each other.",
    },
    "regulatory_filing": {
        "strategy": "filing_section_boundary",
        "max_tokens": 220,
        "min_overlap": 24,
        "rationale": "Filings require section-local context for disclosed facts, risks, dates, and scoped assertions.",
    },
    "social_media_post": {
        "strategy": "thread_or_post_boundary",
        "max_tokens": 120,
        "force_overlap": 0,
        "rationale": "Public-opinion records should preserve post/thread units and avoid spreading unsafe text across neighbors.",
    },
    "forum_comment": {
        "strategy": "thread_or_post_boundary",
        "max_tokens": 140,
        "force_overlap": 0,
        "rationale": "Forum records are expression evidence; keeping comment boundaries reduces context contamination.",
    },
    "note": {
        "strategy": "conservative_boundary",
        "max_tokens": 160,
        "rationale": "User notes get conservative chunks until source, context, and provenance are reviewed.",
    },
    "transcript": {
        "strategy": "turn_boundary_candidate",
        "max_tokens": 180,
        "min_overlap": 20,
        "rationale": "Transcript chunks should preserve speaker turns and question-answer context where extraction supports it.",
    },
    "archival_primary_source": {
        "strategy": "ocr_conservative_boundary",
        "max_tokens": 90,
        "min_overlap": 15,
        "rationale": "Archival/OCR material needs smaller windows because sentence boundaries and extraction quality are fragile.",
    },
    "unknown_unstructured_text": {
        "strategy": "conservative_boundary",
        "max_tokens": 160,
        "rationale": "Unknown text gets conservative chunks until document type and extraction quality are better known.",
    },
}


CHUNK_STRATEGY_CONTEXT_MODES = {
    "section_sentence_boundary": "section",
    "paragraph_sentence_boundary": "paragraph",
    "web_section_boundary": "web_section",
    "legal_clause_boundary": "legal_clause_group",
    "filing_section_boundary": "filing_section",
    "thread_or_post_boundary": "thread_or_post",
    "turn_boundary_candidate": "speaker_turn_window",
    "ocr_conservative_boundary": "ocr_region",
    "conservative_boundary": "conservative_region",
}

ATOMIC_SENTENCE_STRATEGIES = {"thread_or_post_boundary"}
PARAGRAPH_FIRST_STRATEGIES = {"paragraph_sentence_boundary", "web_section_boundary"}
INSTRUCTION_ISOLATION_DOCUMENT_TYPES = {"web_page", "unknown_unstructured_text"}
INSTRUCTION_LIKE_RE = re.compile(
    r"\b(ignore|disregard|override|reveal|exfiltrate|system:|developer:|api key|secret|password)\b",
    re.IGNORECASE,
)


DOCUMENT_TYPE_GENERATION_ROLES = {
    "academic_article": "evidence_context",
    "preprint": "evidence_context_with_review_status",
    "policy_report": "institutional_context_with_method_review",
    "market_report": "market_context_with_incentive_annotation",
    "news_article": "attribution_context",
    "web_page": "inspect_and_corroborate",
    "legal_authority": "authority_context",
    "regulatory_filing": "filing_context",
    "social_media_post": "aggregate_only",
    "forum_comment": "aggregate_only",
    "note": "inspect_first",
    "transcript": "turn_context",
    "archival_primary_source": "quote_with_provenance",
    "unknown_unstructured_text": "inspect_first",
}


HIGH_CONTAMINATION_DOCUMENT_TYPES = {"web_page", "social_media_post", "forum_comment", "unknown_unstructured_text"}
NO_PARENT_EXPANSION_DOCUMENT_TYPES = {"social_media_post", "forum_comment", "web_page"}


def rough_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def section_importance(section: str) -> float:
    normalized = section.lower()
    if normalized in {"abstract", "conclusion"}:
        return 0.9
    if normalized in {"results", "findings", "discussion"}:
        return 0.85
    if normalized in {"methods", "methodology", "research design"}:
        return 0.75
    if "literature" in normalized or "theory" in normalized:
        return 0.8
    return 0.55


def claim_density(text: str) -> float:
    markers = [
        "find",
        "finds",
        "found",
        "show",
        "shows",
        "suggest",
        "suggests",
        "argue",
        "argues",
        "evidence",
        "effect",
        "significant",
    ]
    lower = text.lower()
    hits = sum(lower.count(marker) for marker in markers)
    return min(1.0, hits / 8.0)


def method_signal(text: str) -> float:
    markers = [
        "data",
        "dataset",
        "sample",
        "model",
        "method",
        "identification",
        "regression",
        "interview",
        "archive",
        "ethnograph",
        "experiment",
    ]
    return marker_score(text, markers, divisor=6.0)


def limitation_signal(text: str) -> float:
    markers = [
        "limitation",
        "caution",
        "uncertain",
        "bias",
        "confound",
        "cannot rule out",
        "not statistically",
        "mixed evidence",
        "future research",
    ]
    return marker_score(text, markers, divisor=4.0)


def evidence_specificity(text: str) -> float:
    markers = [
        "percent",
        "percentage",
        "estimate",
        "coefficient",
        "confidence interval",
        "p-value",
        "case study",
        "from 19",
        "from 20",
        "n=",
        "respondents",
        "participants",
    ]
    return marker_score(text, markers, divisor=5.0)


def uncertainty_signal(text: str) -> float:
    markers = [
        "may",
        "might",
        "suggest",
        "suggests",
        "uncertain",
        "mixed",
        "heterogeneous",
        "conditional",
        "depends",
    ]
    return marker_score(text, markers, divisor=6.0)


def marker_score(text: str, markers: list[str], divisor: float) -> float:
    lower = text.lower()
    hits = sum(lower.count(marker) for marker in markers)
    return min(1.0, hits / divisor)


def evidence_role_vector(section: str, text: str) -> dict[str, float]:
    return {
        "section_role": section_importance(section),
        "claim_density": claim_density(text),
        "method_signal": method_signal(text),
        "limitation_signal": limitation_signal(text),
        "evidence_specificity": evidence_specificity(text),
        "uncertainty_signal": uncertainty_signal(text),
    }


def chunk_importance(
    section: str,
    text: str,
    start: int,
    end: int,
    total_tokens: int,
    boundary_aligned: bool,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    vector = evidence_role_vector(section, text)
    token_count = max(0, end - start)
    policy = policy or {}
    decision_profile = chunk_decision_profile(policy, token_count, boundary_aligned)
    vector.update(
        {
            "chunk_token_count": float(token_count),
            "chunk_relative_start": round(start / total_tokens, 6) if total_tokens else 0.0,
            "chunk_relative_end": round(end / total_tokens, 6) if total_tokens else 0.0,
            "boundary_aligned": 1.0 if boundary_aligned else 0.0,
            "chunk_strategy": policy.get("strategy", "fixed_token_window"),
            "chunk_policy_document_type": policy.get("document_type", "unknown"),
            "chunk_policy_domain": policy.get("domain", "unknown"),
            "chunk_policy_target_tokens": float(policy.get("chunk_tokens", 0.0) or 0.0),
            "chunk_policy_overlap_tokens": float(policy.get("overlap_tokens", 0.0) or 0.0),
            **decision_profile,
        }
    )
    return vector


def chunk_decision_profile(policy: dict[str, Any], token_count: int, boundary_aligned: bool) -> dict[str, Any]:
    document_type = str(policy.get("document_type") or "unknown_unstructured_text")
    strategy = str(policy.get("strategy") or "fixed_token_window")
    target_tokens = float(policy.get("chunk_tokens", 0.0) or 0.0)
    contamination_risk = chunk_safety_contamination_risk(document_type, boundary_aligned)
    containment_score = evidence_containment_score(token_count, target_tokens, boundary_aligned, strategy)
    safe_to_expand = (
        boundary_aligned
        and contamination_risk < 0.65
        and document_type not in NO_PARENT_EXPANSION_DOCUMENT_TYPES
    )
    return {
        "chunk_resolution": chunk_resolution(strategy, document_type),
        "parent_context_mode": CHUNK_STRATEGY_CONTEXT_MODES.get(strategy, "token_window"),
        "parent_context_recommended": 1.0 if safe_to_expand else 0.0,
        "generation_context_role": DOCUMENT_TYPE_GENERATION_ROLES.get(document_type, "inspect_first"),
        "context_expansion_policy": context_expansion_policy(document_type, safe_to_expand),
        "evidence_containment_score": containment_score,
        "chunk_safety_contamination_risk": contamination_risk,
        "chunk_quality_tier": chunk_quality_tier(containment_score, contamination_risk),
    }


def chunk_resolution(strategy: str, document_type: str) -> str:
    if document_type in {"social_media_post", "forum_comment"}:
        return "atomic_public_expression"
    if strategy in {"legal_clause_boundary", "thread_or_post_boundary", "turn_boundary_candidate"}:
        return "atomic_boundary_unit"
    if strategy in {"section_sentence_boundary", "filing_section_boundary"}:
        return "evidence_window"
    if strategy in {"ocr_conservative_boundary", "conservative_boundary"}:
        return "conservative_window"
    return "retrieval_window"


def context_expansion_policy(document_type: str, safe_to_expand: bool) -> str:
    if document_type in {"social_media_post", "forum_comment"}:
        return "no_parent_expansion_aggregate_only"
    if document_type == "web_page":
        return "no_parent_expansion_until_safety_and_corroboration"
    if document_type == "unknown_unstructured_text":
        return "inspect_before_parent_expansion"
    if safe_to_expand:
        return "safe_parent_expansion_allowed"
    return "keep_atomic_until_boundary_review"


def chunk_safety_contamination_risk(document_type: str, boundary_aligned: bool) -> float:
    if document_type in {"social_media_post", "forum_comment"}:
        base = 0.8
    elif document_type == "web_page":
        base = 0.7
    elif document_type == "unknown_unstructured_text":
        base = 0.6
    elif document_type in {"archival_primary_source", "transcript"}:
        base = 0.4
    else:
        base = 0.25
    if boundary_aligned:
        base -= 0.15
    else:
        base += 0.15
    return round(min(1.0, max(0.0, base)), 6)


def evidence_containment_score(token_count: int, target_tokens: float, boundary_aligned: bool, strategy: str) -> float:
    if token_count <= 0:
        return 0.0
    boundary_score = 0.65 if boundary_aligned else 0.25
    if target_tokens <= 0:
        size_score = 0.4
    else:
        ratio = token_count / target_tokens
        if 0.35 <= ratio <= 1.05:
            size_score = 0.25
        elif ratio < 0.35:
            size_score = 0.15
        else:
            size_score = 0.05
    strategy_score = 0.1 if strategy != "fixed_token_window" else 0.0
    return round(min(1.0, boundary_score + size_score + strategy_score), 6)


def chunk_quality_tier(containment_score: float, contamination_risk: float) -> str:
    if contamination_risk >= 0.75:
        return "safety_sensitive"
    if containment_score >= 0.85 and contamination_risk < 0.5:
        return "strong_boundary_context"
    if containment_score >= 0.6:
        return "usable_with_review"
    return "weak_boundary_or_size_fit"


def chunking_policy(work: Work, section: str, chunk_tokens: int, overlap_tokens: int) -> dict[str, Any]:
    raw = work.raw or {}
    raw_document_type = normalized_value(raw.get("document_type") or raw.get("type") or raw.get("source_type"))
    if raw_document_type in DOCUMENT_TYPE_CHUNK_POLICIES:
        document_type = raw_document_type
    else:
        profiled_document_type = document_type_profile(work).document_type
        document_type = profiled_document_type if profiled_document_type != "unknown_unstructured_text" or raw_document_type else ""
    if not document_type:
        document_type = inferred_document_type(work)
    domain = normalized_value(raw.get("domain") or raw.get("topic_domain"))
    policy = DOCUMENT_TYPE_CHUNK_POLICIES.get(document_type) or DOCUMENT_TYPE_CHUNK_POLICIES["unknown_unstructured_text"]
    target_tokens = min(int(chunk_tokens), int(policy.get("max_tokens", chunk_tokens)))
    if policy.get("force_overlap") is not None:
        overlap = int(policy["force_overlap"])
    else:
        overlap = int(overlap_tokens)
        if policy.get("min_overlap") is not None:
            overlap = max(overlap, int(policy["min_overlap"]))
    if target_tokens <= 0:
        target_tokens = max(1, int(chunk_tokens))
    if overlap >= target_tokens:
        overlap = max(0, target_tokens // 4)
    return {
        "strategy": policy["strategy"],
        "rationale": policy["rationale"],
        "document_type": document_type or "unknown",
        "domain": domain or "unknown",
        "section": section,
        "chunk_tokens": target_tokens,
        "overlap_tokens": overlap,
        "requested_chunk_tokens": int(chunk_tokens),
        "requested_overlap_tokens": int(overlap_tokens),
    }


def normalized_value(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def inferred_document_type(work: Work) -> str:
    source = str(work.source_name or "").strip().lower()
    academic_source_markers = ["journal", "university press", "academic press", "proceedings", "conference"]
    if work.doi or any(marker in source for marker in academic_source_markers):
        return "academic_article"
    return "unknown_unstructured_text"


def chunk_text(
    work: Work,
    section: str,
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    policy = chunking_policy(work, section, chunk_tokens, overlap_tokens)
    chunk_tokens = int(policy["chunk_tokens"])
    overlap_tokens = int(policy["overlap_tokens"])
    tokens = rough_tokens(text)
    if not tokens:
        return []
    sentences = sentence_spans(text)
    strategy = str(policy.get("strategy") or "")
    document_type = str(policy.get("document_type") or "")
    if strategy in ATOMIC_SENTENCE_STRATEGIES and sentences:
        return atomic_sentence_chunks(work, section, sentences, len(tokens), chunk_tokens, policy)
    if document_type in INSTRUCTION_ISOLATION_DOCUMENT_TYPES and contains_instruction_like_sentence(sentences):
        return atomic_sentence_chunks(work, section, sentences, len(tokens), chunk_tokens, policy)
    if strategy in PARAGRAPH_FIRST_STRATEGIES:
        paragraphs = paragraph_spans(text)
        if len(paragraphs) > 1:
            return paragraph_first_chunks(work, section, paragraphs, len(tokens), chunk_tokens, overlap_tokens, policy)
    if len(sentences) > 1:
        return boundary_chunks(work, section, sentences, len(tokens), chunk_tokens, overlap_tokens, policy)
    if len(sentences) == 1 and len(tokens) <= chunk_tokens and text.strip().endswith((".", "!", "?")):
        return [make_chunk(work, section, text.strip(), 0, len(tokens), len(tokens), boundary_aligned=True, policy=policy)]
    return sliding_token_chunks(work, section, tokens, 0, chunk_tokens, overlap_tokens, boundary_aligned=False, policy=policy)


def sentence_spans(text: str) -> list[tuple[str, int, int]]:
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text.strip()) if part.strip()]
    spans = []
    start = 0
    for part in parts:
        length = len(rough_tokens(part))
        if length:
            spans.append((part, start, start + length))
            start += length
    return spans


def contains_instruction_like_sentence(sentences: list[tuple[str, int, int]]) -> bool:
    return any(INSTRUCTION_LIKE_RE.search(sentence_text) for sentence_text, _, _ in sentences)


def paragraph_spans(text: str) -> list[tuple[str, int, int]]:
    parts = [part.strip() for part in re.split(r"\n\s*\n+", text.strip()) if part.strip()]
    spans = []
    start = 0
    for part in parts:
        length = len(rough_tokens(part))
        if length:
            spans.append((part, start, start + length))
            start += length
    return spans


def atomic_sentence_chunks(
    work: Work,
    section: str,
    sentences: list[tuple[str, int, int]],
    total_tokens: int,
    chunk_tokens: int,
    policy: dict[str, Any],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sentence_text, sentence_start, sentence_end in sentences:
        sentence_len = sentence_end - sentence_start
        if sentence_len > chunk_tokens:
            chunks.extend(
                sliding_token_chunks(
                    work,
                    section,
                    rough_tokens(sentence_text),
                    sentence_start,
                    chunk_tokens,
                    0,
                    boundary_aligned=False,
                    total_tokens=total_tokens,
                    policy=policy_with_parent_span(policy, sentence_start, sentence_end),
                )
            )
        else:
            chunks.append(
                make_chunk(
                    work,
                    section,
                    sentence_text,
                    sentence_start,
                    sentence_end,
                    total_tokens,
                    boundary_aligned=True,
                    policy=policy_with_parent_span(policy, sentence_start, sentence_end),
                )
            )
    return chunks


def paragraph_first_chunks(
    work: Work,
    section: str,
    paragraphs: list[tuple[str, int, int]],
    total_tokens: int,
    chunk_tokens: int,
    overlap_tokens: int,
    policy: dict[str, Any],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for paragraph_text, paragraph_start, paragraph_end in paragraphs:
        paragraph_len = paragraph_end - paragraph_start
        if paragraph_len <= chunk_tokens:
            chunks.append(
                make_chunk(
                    work,
                    section,
                    paragraph_text,
                    paragraph_start,
                    paragraph_end,
                    total_tokens,
                    boundary_aligned=True,
                    policy=policy_with_parent_span(policy, paragraph_start, paragraph_end),
                )
            )
            continue
        sentence_chunks = boundary_chunks(
            work,
            section,
            offset_spans(sentence_spans(paragraph_text), paragraph_start),
            total_tokens,
            chunk_tokens,
            overlap_tokens,
            policy_with_parent_span(policy, paragraph_start, paragraph_end),
        )
        if sentence_chunks:
            chunks.extend(sentence_chunks)
        else:
            chunks.extend(
                sliding_token_chunks(
                    work,
                    section,
                    rough_tokens(paragraph_text),
                    paragraph_start,
                    chunk_tokens,
                    overlap_tokens,
                    boundary_aligned=False,
                    total_tokens=total_tokens,
                    policy=policy_with_parent_span(policy, paragraph_start, paragraph_end),
                )
            )
    return chunks


def policy_with_parent_span(policy: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    scoped = dict(policy)
    scoped["parent_context_token_start"] = int(start)
    scoped["parent_context_token_end"] = int(end)
    return scoped


def offset_spans(spans: list[tuple[str, int, int]], offset: int) -> list[tuple[str, int, int]]:
    return [(text, start + offset, end + offset) for text, start, end in spans]


def boundary_chunks(
    work: Work,
    section: str,
    sentences: list[tuple[str, int, int]],
    total_tokens: int,
    chunk_tokens: int,
    overlap_tokens: int,
    policy: dict[str, Any],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[tuple[str, int, int]] = []
    current_len = 0

    def flush_current() -> None:
        if not current:
            return
        start = current[0][1]
        end = current[-1][2]
        value = " ".join(sentence[0] for sentence in current)
        chunks.append(make_chunk(work, section, value, start, end, total_tokens, boundary_aligned=True, policy=policy))

    for sentence in sentences:
        sentence_text, sentence_start, sentence_end = sentence
        sentence_len = sentence_end - sentence_start
        if sentence_len > chunk_tokens:
            flush_current()
            current = []
            current_len = 0
            sentence_tokens = rough_tokens(sentence_text)
            chunks.extend(
                sliding_token_chunks(
                    work,
                    section,
                    sentence_tokens,
                    sentence_start,
                    chunk_tokens,
                    overlap_tokens,
                    boundary_aligned=False,
                    total_tokens=total_tokens,
                    policy=policy,
                )
            )
            continue
        if current and current_len + sentence_len > chunk_tokens:
            flush_current()
            current = overlap_tail(current, overlap_tokens)
            current_len = span_length(current)
            if current and current_len + sentence_len > chunk_tokens:
                current = []
                current_len = 0
        current.append(sentence)
        current_len += sentence_len
    flush_current()
    return chunks


def overlap_tail(sentences: list[tuple[str, int, int]], overlap_tokens: int) -> list[tuple[str, int, int]]:
    if overlap_tokens <= 0:
        return []
    tail: list[tuple[str, int, int]] = []
    total = 0
    for sentence in reversed(sentences):
        length = sentence[2] - sentence[1]
        if total + length > overlap_tokens:
            break
        tail.append(sentence)
        total += length
    return list(reversed(tail))


def span_length(sentences: list[tuple[str, int, int]]) -> int:
    return sum(sentence[2] - sentence[1] for sentence in sentences)


def sliding_token_chunks(
    work: Work,
    section: str,
    tokens: list[str],
    base_start: int,
    chunk_tokens: int,
    overlap_tokens: int,
    boundary_aligned: bool,
    total_tokens: int | None = None,
    policy: dict[str, Any] | None = None,
) -> list[Chunk]:
    total = total_tokens or len(tokens)
    step = max(1, chunk_tokens - overlap_tokens)
    chunks: list[Chunk] = []
    for local_start in range(0, len(tokens), step):
        local_end = min(len(tokens), local_start + chunk_tokens)
        start = base_start + local_start
        end = base_start + local_end
        value = " ".join(tokens[local_start:local_end])
        chunks.append(make_chunk(work, section, value, start, end, total, boundary_aligned=boundary_aligned, policy=policy))
        if local_end == len(tokens):
            break
    return chunks


def make_chunk(
    work: Work,
    section: str,
    text: str,
    start: int,
    end: int,
    total_tokens: int,
    boundary_aligned: bool,
    policy: dict[str, Any] | None = None,
) -> Chunk:
    digest = hashlib.sha1(f"{work.id}:{section}:{start}:{end}".encode("utf-8")).hexdigest()[:16]
    chunk_id = f"chunk:{digest}"
    importance = chunk_importance(section, text, start, end, total_tokens, boundary_aligned, policy)
    importance.update(parent_context_metadata(work, section, total_tokens, chunk_id, policy))
    return Chunk(
        id=chunk_id,
        work_id=work.id,
        section=section,
        text=text,
        token_start=start,
        token_end=end,
        importance=importance,
    )


def parent_context_metadata(
    work: Work,
    section: str,
    total_tokens: int,
    chunk_id: str,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = policy or {}
    mode = CHUNK_STRATEGY_CONTEXT_MODES.get(str(policy.get("strategy") or ""), "token_window")
    parent_start = policy.get("parent_context_token_start")
    parent_end = policy.get("parent_context_token_end")
    scoped_parent = parent_start is not None and parent_end is not None
    span_start = int(parent_start) if scoped_parent else 0
    span_end = int(parent_end) if scoped_parent else total_tokens
    parent_id = stable_parent_context_id(work.id, section, mode, span_start, span_end)
    return {
        "atomic_chunk_id": chunk_id,
        "parent_context_id": parent_id,
        "parent_context_token_start": float(span_start),
        "parent_context_token_end": float(span_end),
        "parent_context_span_tokens": float(max(0, span_end - span_start)),
        "parent_context_available": 1.0,
    }


def stable_parent_context_id(work_id: str, section: str, mode: str, start: int, end: int) -> str:
    normalized = f"{work_id}:{section.strip().lower()}:{mode}:{start}:{end}"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"parent:{digest}"


def chunk_work_abstract(work: Work, chunk_tokens: int, overlap_tokens: int) -> list[Chunk]:
    if not work.abstract:
        return []
    return chunk_text(work, "abstract", work.abstract, chunk_tokens, overlap_tokens)
