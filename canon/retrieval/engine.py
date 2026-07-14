from __future__ import annotations

from typing import Sequence

from canon.evidence.committee import committee_candidate_row, corroboration_item, generator_context_allowed
from canon.evidence.corroboration import assess_corroboration
from canon.retrieval.bm25 import BM25Index
from canon.retrieval.corpus import RetrievalDocument
from canon.retrieval.decisions import apply_evidence_decision
from canon.retrieval.policies import components_for_document, normalize, weighted_score
from canon.retrieval.semantic import semantic_scores
from canon.retrieval.stages import (
    apply_safety_gate,
    candidate_pool_size,
    enrich_candidates,
    retrieval_stage_summary,
    retrieve_candidates,
)
from canon.retrieval.trace import RetrievalTraceItem, make_trace_item
from canon.retrieval.tokenize import tokenize


DISAGREEMENT_INTENT_TERMS = {
    "challenge",
    "challenges",
    "conflict",
    "conflicts",
    "contradict",
    "contradicts",
    "contradiction",
    "critique",
    "critiques",
    "debate",
    "debates",
    "disagree",
    "disagreement",
    "disputes",
    "mixed",
    "skeptical",
    "tension",
    "versus",
}

DISAGREEMENT_MIN_DIVERSITY = 0.25
EXCLUSION_MARKERS = {"excluding", "exclude", "except", "without"}
EXCLUSION_STOP_TERMS = {"and", "but", "compared", "rather", "than", "versus", "vs"}
DEFAULT_EXCLUSION_PENALTY = 0.6
CONTROL_WEIGHT_KEYS = {
    "diversity",
    "diversity_focus_floor",
    "diversity_relevance_floor",
    "exclusion_penalty",
    "low_relevance_penalty",
    "method_signal_bonus",
    "limitation_signal_bonus",
    "evidence_specificity_bonus",
    "candidate_k",
    "committee_gate",
    "candidate_syntax_weight",
    "relevance_floor",
}
GENERIC_QUERY_TERMS = {
    "about",
    "answer",
    "corpus",
    "does",
    "evidence",
    "literature",
    "research",
    "says",
    "this",
    "what",
    "where",
    "which",
}

_BM25_INDEX_CACHE: dict[tuple[tuple[str, int, int], ...], BM25Index] = {}
_BM25_SCORE_CACHE: dict[tuple[str, tuple[tuple[str, int, int], ...]], list[float]] = {}
_SEMANTIC_SCORE_CACHE: dict[tuple[str, tuple[tuple[str, int, int], ...]], list[float]] = {}


def retrieve(
    query: str,
    documents: Sequence[RetrievalDocument],
    weights: dict[str, float],
    top_k: int,
    preview_chars: int,
) -> list[RetrievalTraceItem]:
    return retrieve_with_stages(query, documents, weights, top_k, preview_chars)["trace"]


def retrieve_with_stages(
    query: str,
    documents: Sequence[RetrievalDocument],
    weights: dict[str, float],
    top_k: int,
    preview_chars: int,
) -> dict:
    effective_weights = weights_with_query_intent(query, weights)
    requested_candidate_k = int(effective_weights["candidate_k"]) if "candidate_k" in effective_weights else None
    candidates = retrieve_candidates(
        query,
        documents,
        candidate_pool_size(top_k, len(documents), requested_candidate_k),
        candidate_syntax_weight=float(effective_weights.get("candidate_syntax_weight", 0.1) or 0.0),
    )
    enriched = enrich_candidates(query, candidates)
    gated = apply_safety_gate(enriched)
    committee_gate = committee_gate_enabled(effective_weights)
    committee_rows = committee_rows_by_chunk(query, gated, preview_chars) if committee_gate else {}
    scored = []
    decisions = {}
    rejected_candidates = []
    committee_excluded_candidates = []
    for gate_result in gated:
        document = gate_result.enriched.candidate.document
        components = gate_result.enriched.components
        committee_row = committee_rows.get(document.chunk_id)
        stage = {
            "candidate": gate_result.enriched.signals["query_text_relevance"],
            "signals": gate_result.enriched.signals,
            "safety_gate": gate_result.as_dict(),
        }
        if committee_row:
            stage["committee_gate"] = committee_row
        if gate_result.decision != "allow":
            rejected_candidates.append(
                {
                    "chunk_id": document.chunk_id,
                    "work_id": document.work_id,
                    "title": document.title,
                    "stage": stage,
                }
            )
            continue
        if committee_gate and committee_row and not generator_context_allowed(committee_row):
            rejected_row = {
                "chunk_id": document.chunk_id,
                "work_id": document.work_id,
                "title": document.title,
                "committee_decision": committee_row["committee_decision"],
                "allowed_uses": committee_row["allowed_uses"],
                "conflict_tags": committee_row["conflict_tags"],
                "stage": stage,
            }
            rejected_candidates.append(rejected_row)
            committee_excluded_candidates.append(rejected_row)
            continue
        base_score = weighted_score(components, weights_without_diversity(effective_weights))
        base_score = apply_query_controls(
            query,
            document,
            components.relevance,
            base_score,
            effective_weights,
        )
        decision = apply_evidence_decision(
            query,
            document,
            components,
            base_score,
            effective_weights,
            safety=gate_result.safety,
            stage=stage,
        )
        decisions[document.chunk_id] = decision
        if decision.decision == "allow" and decision.score > 0.0:
            scored.append((decision.score, base_score, document, components))
    if not scored:
        return {
            "trace": [],
            "stage_summary": retrieval_stage_summary_with_committee(
                candidates,
                enriched,
                gated,
                committee_rows,
                committee_excluded_candidates,
            ),
            "rejected_candidates": rejected_candidates,
        }
    scored = select_with_diversity(
        scored,
        top_k,
        float(effective_weights.get("diversity", 0.0)),
        query=query,
        weights=effective_weights,
    )
    trace = [
        make_trace_item(
            rank,
            document,
            final,
            components,
            effective_weights,
            preview_chars,
            base_score=base,
            decision=decisions.get(document.chunk_id).to_dict() if document.chunk_id in decisions else {},
        )
        for rank, (final, base, document, components) in enumerate(scored, start=1)
    ]
    return {
        "trace": trace,
        "stage_summary": retrieval_stage_summary_with_committee(
            candidates,
            enriched,
            gated,
            committee_rows,
            committee_excluded_candidates,
        ),
        "rejected_candidates": rejected_candidates,
    }


def committee_gate_enabled(weights: dict[str, float]) -> bool:
    return float(weights.get("committee_gate", 0.0) or 0.0) > 0.0


def committee_rows_by_chunk(
    query: str,
    gated: list,
    preview_chars: int,
) -> dict[str, dict]:
    if not gated:
        return {}
    corroboration = assess_corroboration([corroboration_item(result) for result in gated])
    rows = [
        committee_candidate_row(result, query=query, corroboration=corroboration, preview_chars=preview_chars)
        for result in gated
    ]
    return {row["chunk_id"]: row for row in rows}


def retrieval_stage_summary_with_committee(
    candidates: list,
    enriched: list,
    gated: list,
    committee_rows: dict[str, dict],
    committee_excluded_candidates: list[dict],
) -> dict:
    summary = retrieval_stage_summary(candidates, enriched, gated)
    if not committee_rows:
        return summary
    decision_counts: dict[str, int] = {}
    conflict_counts: dict[str, int] = {}
    for row in committee_rows.values():
        decision = str(row.get("committee_decision", "unknown"))
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        for tag in row.get("conflict_tags") or []:
            conflict_counts[tag] = conflict_counts.get(tag, 0) + 1
    summary.update(
        {
            "committee_gate_enabled": True,
            "committee_evaluated_count": len(committee_rows),
            "committee_excluded_from_context_count": len(committee_excluded_candidates),
            "committee_decision_counts": dict(sorted(decision_counts.items())),
            "committee_conflict_tag_counts": dict(sorted(conflict_counts.items())),
        }
    )
    return summary


def corpus_signature(documents: Sequence[RetrievalDocument]) -> tuple[tuple[str, int, int], ...]:
    return tuple((document.chunk_id, len(document.text), hash(document.text)) for document in documents)


def cached_bm25_scores(
    query: str,
    texts: list[str],
    signature: tuple[tuple[str, int, int], ...],
) -> list[float]:
    key = (query, signature)
    if key not in _BM25_SCORE_CACHE:
        index = _BM25_INDEX_CACHE.get(signature)
        if index is None:
            index = BM25Index(texts)
            _BM25_INDEX_CACHE[signature] = index
        _BM25_SCORE_CACHE[key] = index.scores(query)
    return list(_BM25_SCORE_CACHE[key])


def cached_semantic_scores(
    query: str,
    texts: list[str],
    signature: tuple[tuple[str, int, int], ...],
) -> list[float]:
    key = (query, signature)
    if key not in _SEMANTIC_SCORE_CACHE:
        _SEMANTIC_SCORE_CACHE[key] = semantic_scores(query, texts)
    return list(_SEMANTIC_SCORE_CACHE[key])


def weights_with_query_intent(query: str, weights: dict[str, float]) -> dict[str, float]:
    effective = dict(weights)
    if has_disagreement_intent(query):
        effective["diversity"] = max(float(effective.get("diversity", 0.0)), DISAGREEMENT_MIN_DIVERSITY)
    return effective


def has_disagreement_intent(query: str) -> bool:
    return bool(set(tokenize(query)) & DISAGREEMENT_INTENT_TERMS)


def apply_query_controls(
    query: str,
    document: RetrievalDocument,
    relevance: float,
    score: float,
    weights: dict[str, float],
) -> float:
    adjusted = score
    excluded_terms = excluded_query_terms(query)
    if excluded_terms:
        penalty = float(weights.get("exclusion_penalty", DEFAULT_EXCLUSION_PENALTY))
        adjusted -= penalty * document_term_coverage(document, excluded_terms)
    relevance_floor = float(weights.get("relevance_floor", 0.0))
    if relevance_floor > 0 and relevance < relevance_floor:
        penalty = float(weights.get("low_relevance_penalty", 0.0))
        adjusted -= penalty * ((relevance_floor - relevance) / relevance_floor)
    return max(0.0, adjusted)


def excluded_query_terms(query: str) -> list[str]:
    tokens = tokenize(query)
    excluded: list[str] = []
    collecting = False
    for token in tokens:
        if token in EXCLUSION_MARKERS:
            collecting = True
            continue
        if not collecting:
            continue
        if token in EXCLUSION_STOP_TERMS:
            break
        if len(token) >= 4:
            excluded.append(token)
    return sorted(set(excluded))


def document_term_coverage(document: RetrievalDocument, terms: list[str]) -> float:
    if not terms:
        return 0.0
    document_tokens = set(tokenize(" ".join([document.title, document.text])))
    return len([term for term in terms if term in document_tokens]) / len(terms)


def weights_without_diversity(weights: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in weights.items() if key not in CONTROL_WEIGHT_KEYS}


def select_with_diversity(
    scored: list[tuple[float, float, RetrievalDocument, object]],
    top_k: int,
    diversity_weight: float,
    query: str = "",
    weights: dict[str, float] | None = None,
) -> list[tuple[float, float, RetrievalDocument, object]]:
    remaining = sorted(scored, key=lambda row: (row[0], row[2].year or 0, row[2].title), reverse=True)
    if diversity_weight <= 0:
        return remaining[:top_k]
    weights = weights or {}
    selected: list[tuple[float, float, RetrievalDocument, object]] = []
    used_clusters: set[int] = set()
    while remaining and len(selected) < top_k:
        best_index = 0
        best_score = -1.0
        for index, (final_score, base_score, document, components) in enumerate(remaining):
            cluster_bonus = 0.0
            if (
                document.cluster_id is not None
                and document.cluster_id not in used_clusters
                and diversity_eligible(query, document, components, weights)
            ):
                cluster_bonus = diversity_weight
            adjusted = final_score + cluster_bonus
            if adjusted > best_score:
                best_index = index
                best_score = adjusted
        _final_score, base_score, document, components = remaining.pop(best_index)
        if document.cluster_id is not None:
            used_clusters.add(document.cluster_id)
        selected.append((best_score, base_score, document, components))
    return selected


def diversity_eligible(
    query: str,
    document: RetrievalDocument,
    components: object,
    weights: dict[str, float],
) -> bool:
    relevance_floor = float(weights.get("diversity_relevance_floor", 0.0))
    if relevance_floor > 0:
        relevance = float(getattr(components, "relevance", 0.0))
        if relevance < relevance_floor:
            return False
    focus_floor = float(weights.get("diversity_focus_floor", 0.0))
    if focus_floor > 0 and query_focus_coverage(query, document) < focus_floor:
        return False
    return True


def query_focus_coverage(query: str, document: RetrievalDocument) -> float:
    query_terms = [
        token
        for token in sorted(set(tokenize(query)))
        if len(token) >= 4 and token not in GENERIC_QUERY_TERMS
    ]
    if not query_terms:
        return 0.0
    document_tokens = set(tokenize(" ".join([document.title, document.text])))
    return len([term for term in query_terms if term in document_tokens]) / len(query_terms)
