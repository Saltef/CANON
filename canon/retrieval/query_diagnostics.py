from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from canon.config import load_settings
from canon.retrieval.corpus import RetrievalDocument
from canon.retrieval.experiment import cached_documents
from canon.retrieval.semantic import HashedSemanticEncoder, cosine
from canon.retrieval.stages import retrieve_candidates
from canon.retrieval.tokenize import tokenize


STOP_TERMS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "but",
    "can",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "its",
    "more",
    "not",
    "out",
    "over",
    "that",
    "the",
    "their",
    "them",
    "this",
    "through",
    "under",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
}

FREEDOM_THRESHOLDS = {
    "strict": {"min_support": 2, "min_similarity": 0.2, "max_variants": 3},
    "balanced": {"min_support": 2, "min_similarity": 0.08, "max_variants": 5},
    "exploratory": {"min_support": 1, "min_similarity": 0.03, "max_variants": 8},
}


def diagnose_query(
    query: str,
    documents: Sequence[RetrievalDocument],
    top_k: int = 5,
    candidate_k: int = 20,
    freedom_level: str = "balanced",
    reviewer_patterns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if freedom_level not in FREEDOM_THRESHOLDS:
        raise ValueError(f"Unknown freedom level '{freedom_level}'.")
    candidates = retrieve_candidates(query, documents, min(candidate_k, len(documents)))
    top_candidates = candidates[:top_k]
    query_terms = set(content_terms(query))
    matched_terms = sorted(
        term
        for term in query_terms
        if any(term_matches_tokens(term, candidate_terms(candidate)) for candidate in top_candidates)
    )
    weak_terms = sorted(term for term in query_terms if term not in set(matched_terms))
    phrases = result_neighborhood_phrases(
        query=query,
        candidates=top_candidates,
        freedom_level=freedom_level,
        reviewer_patterns=reviewer_patterns or {},
    )
    variants = query_variants(query, phrases, candidates, documents, freedom_level)
    return {
        "original_query": query,
        "freedom_level": freedom_level,
        "query_to_corpus": {
            "matched_terms": matched_terms,
            "weak_terms": weak_terms,
            "semantic_similarity_summary": semantic_similarity_summary(top_candidates),
            "top_candidate_count": len(top_candidates),
        },
        "result_neighborhood": {
            "field_phrases": [phrase["phrase"] for phrase in phrases],
            "phrase_details": phrases,
            "missing_from_query": [
                phrase["phrase"]
                for phrase in phrases
                if not phrase_contains_query_term(phrase["phrase"], query_terms)
            ],
        },
        "query_variants": variants,
        "stability": stability_summary(variants),
        "limitations": [
            "Query diagnostics are retrieval aids, not proof that a suggested phrase is correct.",
            "Exploratory variants can increase recall while increasing drift risk.",
            "Suggested phrases must be reviewed before they are used to strengthen an answer.",
        ],
    }


def content_terms(text: str) -> list[str]:
    acronym_terms = {
        token.casefold()
        for token in re.findall(r"\b[A-Z]{2,6}\b", text or "")
        if token.casefold() not in STOP_TERMS
    }
    return sorted(
        {
            token
            for token in tokenize(text)
            if (len(token) >= 3 or token in acronym_terms) and token not in STOP_TERMS
        }
    )


def term_matches_tokens(term: str, tokens: set[str]) -> bool:
    if term in tokens:
        return True
    if term.endswith("s") and len(term) > 3 and term[:-1] in tokens:
        return True
    return f"{term}s" in tokens


def candidate_terms(candidate) -> set[str]:
    document = candidate.document
    return set(tokenize(" ".join([document.title or "", document.text or ""])))


def semantic_similarity(left: str, right: str) -> float:
    encoder = HashedSemanticEncoder()
    return cosine(encoder.encode(left), encoder.encode(right))


def semantic_similarity_summary(candidates: list) -> dict[str, float]:
    values = [float(candidate.semantic_similarity) for candidate in candidates]
    if not values:
        return {"top_mean": 0.0, "top_max": 0.0}
    return {
        "top_mean": round(sum(values) / len(values), 6),
        "top_max": round(max(values), 6),
    }


def result_neighborhood_phrases(
    query: str,
    candidates: list,
    freedom_level: str,
    reviewer_patterns: dict[str, Any],
) -> list[dict[str, Any]]:
    thresholds = FREEDOM_THRESHOLDS[freedom_level]
    query_terms = set(content_terms(query))
    accepted = {normalize_phrase(phrase) for phrase in reviewer_patterns.get("accepted_phrases", [])}
    rejected = {normalize_phrase(phrase) for phrase in reviewer_patterns.get("rejected_phrases", [])}
    edited = {normalize_phrase(phrase) for phrase in reviewer_patterns.get("edited_phrases", [])}
    counts: Counter[str] = Counter()
    semantic_mass: Counter[str] = Counter()
    for candidate in candidates:
        for phrase in candidate_phrases(candidate.document.text):
            normalized = normalize_phrase(phrase)
            if normalized in rejected:
                continue
            counts[normalized] += 1
            semantic_mass[normalized] += float(candidate.semantic_similarity)
    rows = []
    for phrase, support in counts.items():
        similarity = semantic_similarity(query, phrase)
        if support < thresholds["min_support"] and phrase not in accepted and phrase not in edited:
            continue
        if similarity < thresholds["min_similarity"] and phrase not in accepted and phrase not in edited:
            continue
        history = reviewer_history(phrase, accepted, rejected, edited)
        rows.append(
            {
                "phrase": phrase,
                "source": "accepted_reviewer_pattern" if history == "accepted" else "semantic_result_neighborhood",
                "support_count": int(support),
                "semantic_distance": round(max(0.0, 1.0 - similarity), 6),
                "mean_candidate_semantic_similarity": round(float(semantic_mass[phrase]) / support, 6),
                "reviewer_history": history,
                "drift_risk": drift_risk(similarity, support, query_terms, phrase),
                "exploration_label": exploration_label(similarity, query_terms, phrase),
            }
        )
    rows.sort(
        key=lambda row: (
            row["reviewer_history"] == "accepted",
            row["support_count"],
            -row["semantic_distance"],
            row["phrase"],
        ),
        reverse=True,
    )
    return rows[: int(thresholds["max_variants"])]


def candidate_phrases(text: str) -> list[str]:
    terms = [token for token in tokenize(text) if len(token) >= 3 and token not in STOP_TERMS]
    phrases: list[str] = []
    for size in (2, 3):
        for index in range(0, max(0, len(terms) - size + 1)):
            phrase = " ".join(terms[index : index + size])
            if len(set(phrase.split())) > 1:
                phrases.append(phrase)
    return phrases


def normalize_phrase(phrase: str) -> str:
    return " ".join(token for token in tokenize(phrase) if len(token) >= 3 and token not in STOP_TERMS)


def reviewer_history(phrase: str, accepted: set[str], rejected: set[str], edited: set[str]) -> str:
    normalized = normalize_phrase(phrase)
    if normalized in accepted:
        return "accepted"
    if normalized in rejected:
        return "rejected"
    if normalized in edited:
        return "edited"
    return "unseen"


def drift_risk(similarity: float, support: int, query_terms: set[str], phrase: str) -> str:
    overlaps_query = phrase_contains_query_term(phrase, query_terms)
    if similarity >= 0.2 and (support >= 2 or overlaps_query):
        return "low"
    if similarity >= 0.08 or support >= 2:
        return "medium"
    return "high"


def exploration_label(similarity: float, query_terms: set[str], phrase: str) -> str:
    if phrase_contains_query_term(phrase, query_terms):
        return "close_match"
    if similarity >= 0.2:
        return "field_synonym"
    if similarity >= 0.08:
        return "adjacent_concept"
    return "exploratory_tangent"


def phrase_contains_query_term(phrase: str, query_terms: set[str]) -> bool:
    phrase_terms = set(content_terms(phrase))
    return bool(phrase_terms & query_terms)


def query_variants(
    query: str,
    phrases: list[dict[str, Any]],
    original_candidates: list,
    documents: Sequence[RetrievalDocument],
    freedom_level: str,
) -> list[dict[str, Any]]:
    original_ids = [candidate.document.chunk_id for candidate in original_candidates[:5]]
    variants = []
    for phrase in phrases:
        variant_query = phrase["phrase"]
        variant_candidates = retrieve_candidates(variant_query, documents, min(20, len(documents)))
        variant_ids = [candidate.document.chunk_id for candidate in variant_candidates[:5]]
        variants.append(
            {
                "query": variant_query,
                "reason": variant_reason(phrase),
                "result_overlap_with_original": rank_overlap(original_ids, variant_ids),
                "freedom_level": freedom_level_for_phrase(freedom_level, phrase),
                "drift_risk": phrase["drift_risk"],
                "exploration_label": phrase["exploration_label"],
                "source": phrase["source"],
            }
        )
    return variants


def variant_reason(phrase: dict[str, Any]) -> str:
    if phrase["reviewer_history"] in {"accepted", "edited"}:
        return "Previously accepted or edited reviewer wording for this topic."
    if phrase["exploration_label"] == "close_match":
        return "Recurring phrase in semantically similar results that overlaps the original query."
    if phrase["exploration_label"] == "field_synonym":
        return "Semantically close field phrase found in the result neighborhood."
    return "Adjacent concept found in the semantic result neighborhood."


def freedom_level_for_phrase(default_level: str, phrase: dict[str, Any]) -> str:
    if phrase["drift_risk"] == "low":
        return "strict" if default_level == "strict" else "balanced"
    if phrase["drift_risk"] == "medium":
        return "balanced" if default_level != "strict" else "strict"
    return "exploratory"


def rank_overlap(left_ids: list[str], right_ids: list[str]) -> float:
    if not left_ids and not right_ids:
        return 1.0
    if not left_ids or not right_ids:
        return 0.0
    return round(len(set(left_ids) & set(right_ids)) / len(set(left_ids) | set(right_ids)), 6)


def stability_summary(variants: list[dict[str, Any]]) -> dict[str, str]:
    if not variants:
        return {"status": "unknown", "note": "No query variants were generated."}
    average_overlap = sum(float(variant["result_overlap_with_original"]) for variant in variants) / len(variants)
    if average_overlap >= 0.65:
        return {"status": "stable", "note": "Suggested wording retrieves a similar evidence set."}
    if average_overlap >= 0.35:
        return {"status": "sensitive", "note": "Alternate field terminology changes some retrieved evidence."}
    return {"status": "unstable", "note": "Suggested wording changes the evidence set materially."}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose how query wording affects semantic retrieval.")
    parser.add_argument("query")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--freedom-level", choices=sorted(FREEDOM_THRESHOLDS), default="balanced")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = diagnose_query(
        query=args.query,
        documents=cached_documents(args.mode),
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        freedom_level=args.freedom_level,
    )
    if args.write_report:
        settings = load_settings()
        slug = "".join(character if character.isalnum() else "-" for character in args.query.lower())[:60].strip("-")
        write_json(settings.reports_dir / f"query_diagnostics_{args.mode}_{slug}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
