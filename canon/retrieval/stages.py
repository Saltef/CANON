from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from canon.retrieval.bm25 import BM25Index
from canon.retrieval.corpus import RetrievalDocument
from canon.retrieval.policies import ScoreComponents, components_for_document, normalize
from canon.retrieval.semantic import semantic_scores
from canon.retrieval.syntax import (
    corpus_syntax_profile,
    document_syntax_vector,
    primary_syntax_category,
    query_syntax_targets,
    syntax_relevance_scores,
)
from canon.retrieval.tokenize import tokenize
from canon.safety.scanner import SafetyScan, scan_text


DEFAULT_CANDIDATE_MULTIPLIER = 10
DEFAULT_MIN_CANDIDATES = 50
DEFAULT_CANDIDATE_SYNTAX_WEIGHT = 0.1
MAX_CANDIDATE_SYNTAX_WEIGHT = 0.3


@dataclass(frozen=True)
class Candidate:
    document: RetrievalDocument
    lexical_relevance: float
    semantic_similarity: float
    syntax_relevance: float
    syntax_profile: dict
    candidate_score: float
    candidate_rank: int


@dataclass(frozen=True)
class EnrichedCandidate:
    candidate: Candidate
    components: ScoreComponents
    signals: dict


@dataclass(frozen=True)
class SafetyGateResult:
    enriched: EnrichedCandidate
    safety: SafetyScan
    decision: str
    reasons: list[str]

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "reasons": self.reasons,
            "safety": self.safety.as_dict(),
        }


def candidate_pool_size(top_k: int, corpus_size: int, requested: int | None = None) -> int:
    if corpus_size <= 0:
        return 0
    if requested is not None:
        return max(1, min(corpus_size, requested))
    return min(corpus_size, max(top_k * DEFAULT_CANDIDATE_MULTIPLIER, DEFAULT_MIN_CANDIDATES))


def retrieve_candidates(
    query: str,
    documents: Sequence[RetrievalDocument],
    candidate_k: int,
    candidate_syntax_weight: float = DEFAULT_CANDIDATE_SYNTAX_WEIGHT,
) -> list[Candidate]:
    if not documents or candidate_k <= 0:
        return []
    texts = [document.text for document in documents]
    lexical_scores = normalize(BM25Index(texts).scores(query))
    semantic_similarity_scores = normalize(semantic_scores(query, texts))
    syntax_scores = syntax_relevance_scores(query, documents)
    syntax_profile = corpus_syntax_profile(documents).as_dict()
    rows = []
    syntax_weight = min(MAX_CANDIDATE_SYNTAX_WEIGHT, max(0.0, float(candidate_syntax_weight or 0.0)))
    lexical_weight = 0.5 - (syntax_weight * 0.5)
    semantic_weight = 0.35 - (syntax_weight * 0.5)
    focus_weight = 0.15
    for document, lexical, semantic, syntax in zip(
        documents,
        lexical_scores,
        semantic_similarity_scores,
        syntax_scores,
        strict=True,
    ):
        focus = query_focus_coverage(query, document)
        candidate_score = round(
            (lexical * lexical_weight)
            + (semantic * semantic_weight)
            + (syntax * syntax_weight)
            + (focus * focus_weight),
            6,
        )
        rows.append((candidate_score, lexical, semantic, syntax, document))
    ranked = sorted(rows, key=lambda row: (row[0], row[1], row[4].year or 0, row[4].title), reverse=True)
    return [
        Candidate(
            document=document,
            lexical_relevance=lexical,
            semantic_similarity=semantic,
            syntax_relevance=syntax,
            syntax_profile=syntax_profile,
            candidate_score=score,
            candidate_rank=rank,
        )
        for rank, (score, lexical, semantic, syntax, document) in enumerate(ranked[:candidate_k], start=1)
    ]


def enrich_candidates(query: str, candidates: list[Candidate]) -> list[EnrichedCandidate]:
    return [enrich_candidate(query, candidate) for candidate in candidates]


def enrich_candidate(query: str, candidate: Candidate) -> EnrichedCandidate:
    document = candidate.document
    focus_coverage = query_focus_coverage(query, document)
    focus_adjusted_relevance = max(candidate.lexical_relevance, focus_coverage * 0.5)
    focus_adjusted_semantic = max(candidate.semantic_similarity, focus_coverage * 0.35)
    syntax_vector = document_syntax_vector(document)
    query_targets = query_syntax_targets(query)
    components = components_for_document(
        document,
        relevance=focus_adjusted_relevance,
        semantic_similarity=focus_adjusted_semantic,
        syntax_relevance=candidate.syntax_relevance,
    )
    signals = {
        "query_text_relevance": {
            "lexical_relevance": round(candidate.lexical_relevance, 6),
            "semantic_similarity": round(candidate.semantic_similarity, 6),
            "syntax_relevance": round(candidate.syntax_relevance, 6),
            "focus_coverage": round(focus_coverage, 6),
            "focus_adjusted_relevance": round(focus_adjusted_relevance, 6),
            "focus_adjusted_semantic_similarity": round(focus_adjusted_semantic, 6),
            "candidate_score": round(candidate.candidate_score, 6),
            "candidate_rank": candidate.candidate_rank,
        },
        "syntax_profile": {
            "document_syntax_category": primary_syntax_category(syntax_vector),
            "document_syntax_vector": syntax_vector,
            "query_syntax_targets": query_targets,
            "corpus_syntax_profile": candidate.syntax_profile,
            "limitations": [
                "Syntax relevance is a deterministic corpus-profile signal, not a validated learned syntax model.",
                "It detects evidence form and discourse role, not factual reliability.",
            ],
        },
        "evidence_role": {
            "section": document.section,
            "section_role": round(document.chunk_importance.get("section_role", 0.0), 6),
            "claim_density": round(document.chunk_importance.get("claim_density", 0.0), 6),
            "method_signal": round(document.chunk_importance.get("method_signal", 0.0), 6),
            "limitation_signal": round(document.chunk_importance.get("limitation_signal", 0.0), 6),
            "evidence_specificity": round(document.chunk_importance.get("evidence_specificity", 0.0), 6),
            "uncertainty_signal": round(document.chunk_importance.get("uncertainty_signal", 0.0), 6),
        },
        "chunk_profile": {
            "chunk_strategy": str(document.chunk_importance.get("chunk_strategy", "fixed_token_window")),
            "chunk_resolution": str(document.chunk_importance.get("chunk_resolution", "retrieval_window")),
            "atomic_chunk_id": str(document.chunk_importance.get("atomic_chunk_id", document.chunk_id)),
            "parent_context_id": str(document.chunk_importance.get("parent_context_id", "")),
            "parent_context_mode": str(document.chunk_importance.get("parent_context_mode", "token_window")),
            "parent_context_recommended": bool(float(document.chunk_importance.get("parent_context_recommended", 0.0) or 0.0)),
            "generation_context_role": str(document.chunk_importance.get("generation_context_role", "inspect_first")),
            "context_expansion_policy": str(document.chunk_importance.get("context_expansion_policy", "keep_atomic_until_boundary_review")),
            "boundary_aligned": round(float(document.chunk_importance.get("boundary_aligned", 0.0) or 0.0), 6),
            "chunk_token_count": round(float(document.chunk_importance.get("chunk_token_count", 0.0) or 0.0), 6),
            "evidence_containment_score": round(float(document.chunk_importance.get("evidence_containment_score", 0.0) or 0.0), 6),
            "chunk_safety_contamination_risk": round(float(document.chunk_importance.get("chunk_safety_contamination_risk", 0.0) or 0.0), 6),
            "chunk_quality_tier": str(document.chunk_importance.get("chunk_quality_tier", "weak_boundary_or_size_fit")),
        },
        "source_trust": {
            "source_trust": round(document.work_signals.get("source_trust", document.work_signals.get("source_quality", 0.0)), 6),
            "source_quality_legacy": round(document.work_signals.get("source_quality", 0.0), 6),
            "integrity_gate": bool(float(document.work_signals.get("integrity_gate", 1.0))),
            "author_score": round(document.work_signals.get("author_score", 0.0), 6),
            "citation_centrality": round(document.work_signals.get("citation_centrality", 0.0), 6),
        },
        "provenance": {
            "provenance": str(document.work_signals.get("provenance", "academic")),
            "domain": str(document.work_signals.get("domain", "general_academic")),
            "work_id": document.work_id,
            "source_name": document.source_name,
            "year": document.year,
            "cluster_id": document.cluster_id,
        },
        "document_profile": {
            "document_type": str(document.work_signals.get("document_type", "unknown_unstructured_text")),
            "evidence_family": str(document.work_signals.get("evidence_family", "unknown")),
            "document_type_confidence": round(float(document.work_signals.get("document_type_confidence", 0.0)), 6),
            "authority_signal": round(float(document.work_signals.get("authority_signal", 0.0)), 6),
            "timeliness_need": round(float(document.work_signals.get("timeliness_need", 0.0)), 6),
            "source_intent_risk": round(float(document.work_signals.get("source_intent_risk", 0.0)), 6),
            "adversarial_surface": round(float(document.work_signals.get("adversarial_surface", 0.0)), 6),
            "extraction_risk": round(float(document.work_signals.get("extraction_risk", 0.0)), 6),
            "grey_literature": bool(document.work_signals.get("grey_literature", False)),
            "recommended_decision_role": str(document.work_signals.get("recommended_decision_role", "annotate_missing_type")),
        },
        "domain_profile": {
            "domain": str(document.work_signals.get("domain", "general_academic")),
            "domain_profile": str(document.work_signals.get("domain_profile", "general_academic")),
            "domain_fit_score": round(float(document.work_signals.get("domain_fit_score", 0.0)), 6),
            "domain_fit_tier": str(document.work_signals.get("domain_fit_tier", "low_fit")),
            "document_type_fit": round(float(document.work_signals.get("domain_document_type_fit", 0.0)), 6),
            "evidence_family_fit": round(float(document.work_signals.get("domain_evidence_family_fit", 0.0)), 6),
            "authority_fit": round(float(document.work_signals.get("domain_authority_fit", 0.0)), 6),
            "transparency_fit": round(float(document.work_signals.get("domain_transparency_fit", 0.0)), 6),
            "provenance_fit": round(float(document.work_signals.get("domain_provenance_fit", 0.0)), 6),
            "safety_fit": round(float(document.work_signals.get("domain_safety_fit", 0.0)), 6),
            "gate_flags": list(document.work_signals.get("domain_gate_flags", [])),
            "critical_quality_signals": list(document.work_signals.get("domain_critical_quality_signals", [])),
            "required_gates": list(document.work_signals.get("domain_required_gates", [])),
            "annotation_only_signals": list(document.work_signals.get("domain_annotation_only_signals", [])),
            "recency_policy": str(document.work_signals.get("domain_recency_policy", "")),
            "corroboration_policy": str(document.work_signals.get("domain_corroboration_policy", "")),
        },
    }
    return EnrichedCandidate(candidate=candidate, components=components, signals=signals)


def apply_safety_gate(enriched: list[EnrichedCandidate]) -> list[SafetyGateResult]:
    results = []
    for item in enriched:
        document = item.candidate.document
        provenance = str(document.work_signals.get("provenance", "academic"))
        safety = scan_text(document.text, provenance=provenance)
        if safety.decision in {"block", "quarantine"}:
            decision = "reject"
            reasons = [f"safety_{safety.decision}", *safety.categories]
        else:
            decision = "allow"
            reasons = [f"safety_{safety.decision}"] if safety.decision == "sanitize" else []
        results.append(
            SafetyGateResult(
                enriched=item,
                safety=safety,
                decision=decision,
                reasons=reasons,
            )
        )
    return results


def retrieval_stage_summary(
    candidates: list[Candidate],
    enriched: list[EnrichedCandidate],
    gated: list[SafetyGateResult],
) -> dict:
    safety_counts: dict[str, int] = {}
    gate_counts: dict[str, int] = {}
    for result in gated:
        safety_counts[result.safety.decision] = safety_counts.get(result.safety.decision, 0) + 1
        gate_counts[result.decision] = gate_counts.get(result.decision, 0) + 1
    return {
        "candidate_count": len(candidates),
        "enriched_count": len(enriched),
        "allowed_after_safety_gate": gate_counts.get("allow", 0),
        "rejected_after_safety_gate": gate_counts.get("reject", 0),
        "safety_decision_counts": dict(sorted(safety_counts.items())),
    }


def query_focus_coverage(query: str, document: RetrievalDocument) -> float:
    query_terms = [
        token
        for token in sorted(set(tokenize(query)))
        if len(token) >= 4 and token not in {"about", "answer", "corpus", "does", "evidence", "literature", "research", "says", "this", "what", "where", "which"}
    ]
    if not query_terms:
        return 0.0
    document_tokens = set(tokenize(" ".join([document.title, document.text])))
    return len([term for term in query_terms if term in document_tokens]) / len(query_terms)
