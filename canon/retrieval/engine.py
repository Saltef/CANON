from __future__ import annotations

from typing import Sequence

from canon.retrieval.bm25 import BM25Index
from canon.retrieval.corpus import RetrievalDocument
from canon.retrieval.policies import components_for_document, normalize, weighted_score
from canon.retrieval.semantic import semantic_scores
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
    "exclusion_penalty",
    "low_relevance_penalty",
    "relevance_floor",
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
    effective_weights = weights_with_query_intent(query, weights)
    texts = [document.text for document in documents]
    signature = corpus_signature(documents)
    relevance_scores = normalize(cached_bm25_scores(query, texts, signature))
    semantic_similarity_scores = normalize(cached_semantic_scores(query, texts, signature))
    scored = []
    for document, relevance, semantic_similarity in zip(
        documents,
        relevance_scores,
        semantic_similarity_scores,
        strict=True,
    ):
        components = components_for_document(document, relevance, semantic_similarity)
        base_score = weighted_score(components, weights_without_diversity(effective_weights))
        base_score = apply_query_controls(query, document, relevance, base_score, effective_weights)
        scored.append((base_score, base_score, document, components))
    scored = select_with_diversity(scored, top_k, float(effective_weights.get("diversity", 0.0)))
    return [
        make_trace_item(rank, document, final, components, effective_weights, preview_chars, base_score=base)
        for rank, (final, base, document, components) in enumerate(scored, start=1)
    ]


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
) -> list[tuple[float, float, RetrievalDocument, object]]:
    remaining = sorted(scored, key=lambda row: (row[0], row[2].year or 0, row[2].title), reverse=True)
    if diversity_weight <= 0:
        return remaining[:top_k]
    selected: list[tuple[float, float, RetrievalDocument, object]] = []
    used_clusters: set[int] = set()
    while remaining and len(selected) < top_k:
        best_index = 0
        best_score = -1.0
        for index, (final_score, base_score, document, components) in enumerate(remaining):
            cluster_bonus = 0.0
            if document.cluster_id is not None and document.cluster_id not in used_clusters:
                cluster_bonus = diversity_weight
            adjusted = base_score + cluster_bonus
            if adjusted > best_score:
                best_index = index
                best_score = adjusted
        _final_score, base_score, document, components = remaining.pop(best_index)
        if document.cluster_id is not None:
            used_clusters.add(document.cluster_id)
        selected.append((best_score, base_score, document, components))
    return selected
