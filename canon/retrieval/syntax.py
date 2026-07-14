from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from canon.retrieval.corpus import RetrievalDocument
from canon.retrieval.tokenize import tokenize


SYNTAX_CATEGORIES = (
    "method_detail",
    "limitation_caveat",
    "quantitative_result",
    "legal_authority",
    "public_expression",
    "dialogue_or_transcript",
    "risk_disclosure",
    "narrative_context",
    "definition_or_taxonomy",
    "generic_claim",
)

QUERY_TERMS = {
    "method_detail": {
        "method",
        "methods",
        "methodology",
        "sample",
        "sampling",
        "data",
        "dataset",
        "measure",
        "measurement",
        "model",
        "design",
        "identification",
        "experiment",
        "regression",
        "estimate",
    },
    "limitation_caveat": {
        "limitation",
        "limitations",
        "caveat",
        "caveats",
        "bias",
        "uncertainty",
        "weakness",
        "threat",
        "validity",
        "confound",
    },
    "quantitative_result": {
        "number",
        "numbers",
        "share",
        "percent",
        "percentage",
        "rate",
        "effect",
        "estimate",
        "result",
        "results",
        "revenue",
        "market",
        "price",
        "growth",
    },
    "legal_authority": {
        "legal",
        "law",
        "court",
        "case",
        "statute",
        "regulation",
        "filing",
        "jurisdiction",
        "contract",
        "clause",
        "authority",
    },
    "public_expression": {
        "public",
        "opinion",
        "comments",
        "comment",
        "people",
        "residents",
        "community",
        "social",
        "forum",
        "sentiment",
        "stance",
    },
    "dialogue_or_transcript": {
        "transcript",
        "call",
        "speaker",
        "said",
        "statement",
        "interview",
        "hearing",
        "question",
        "answer",
    },
    "risk_disclosure": {
        "risk",
        "risks",
        "exposure",
        "liability",
        "forward",
        "looking",
        "disclosure",
        "uncertain",
        "material",
        "adverse",
    },
    "narrative_context": {
        "history",
        "historical",
        "archive",
        "archival",
        "fieldwork",
        "ethnography",
        "culture",
        "cultural",
        "anthropology",
        "narrative",
    },
    "definition_or_taxonomy": {
        "define",
        "definition",
        "means",
        "category",
        "categories",
        "taxonomy",
        "type",
        "classification",
        "concept",
    },
    "generic_claim": {
        "claim",
        "claims",
        "argument",
        "evidence",
        "finding",
        "findings",
        "conclusion",
    },
}

DOCUMENT_TERMS = {
    "method_detail": QUERY_TERMS["method_detail"] | {"robustness", "variable", "coded", "procedure"},
    "limitation_caveat": QUERY_TERMS["limitation_caveat"] | {"however", "although", "cannot", "may", "might"},
    "quantitative_result": QUERY_TERMS["quantitative_result"] | {"significant", "coefficient", "mean", "median"},
    "legal_authority": QUERY_TERMS["legal_authority"] | {"plaintiff", "defendant", "holding", "section"},
    "public_expression": QUERY_TERMS["public_expression"] | {"post", "thread", "tweet", "reply"},
    "dialogue_or_transcript": QUERY_TERMS["dialogue_or_transcript"] | {"analyst", "operator", "q", "a"},
    "risk_disclosure": QUERY_TERMS["risk_disclosure"] | {"cautionary", "loss", "losses", "impairment"},
    "narrative_context": QUERY_TERMS["narrative_context"] | {"memoir", "oral", "chronicle"},
    "definition_or_taxonomy": QUERY_TERMS["definition_or_taxonomy"] | {"refers", "consists", "class"},
    "generic_claim": QUERY_TERMS["generic_claim"] | {"suggests", "shows", "therefore"},
}

DOCUMENT_TYPE_CATEGORY = {
    "court_opinion": "legal_authority",
    "legal_authority": "legal_authority",
    "regulatory_filing": "risk_disclosure",
    "market_report": "quantitative_result",
    "social_media_post": "public_expression",
    "forum_thread": "public_expression",
    "public_comment": "public_expression",
    "transcript": "dialogue_or_transcript",
    "earnings_call_transcript": "dialogue_or_transcript",
    "interview_transcript": "dialogue_or_transcript",
    "archival_record": "narrative_context",
    "historical_archive": "narrative_context",
}

EVIDENCE_FAMILY_CATEGORY = {
    "public_opinion": "public_expression",
    "official_record": "legal_authority",
    "legal_authority": "legal_authority",
    "market_data": "quantitative_result",
    "interview": "dialogue_or_transcript",
    "archive": "narrative_context",
}

QUANT_PATTERN = re.compile(r"(?:[$€£]\s*\d|\d+(?:\.\d+)?\s*(?:%|percent|bps|million|billion|trillion|x)\b)", re.I)
SPEAKER_PATTERN = re.compile(r"(^|\n)\s*(?:q|a|analyst|operator|speaker|interviewer|respondent|cfo|ceo)\s*[:\-]", re.I)
DEFINITION_PATTERN = re.compile(r"\b(?:is defined as|are defined as|refers to|means|consists of)\b", re.I)


@dataclass(frozen=True)
class SyntaxProfile:
    category_counts: dict[str, int]
    category_weights: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "category_counts": dict(sorted(self.category_counts.items())),
            "category_weights": {key: round(value, 6) for key, value in sorted(self.category_weights.items())},
        }


def query_syntax_targets(query: str) -> dict[str, float]:
    tokens = set(tokenize(query))
    targets = {category: 0.0 for category in SYNTAX_CATEGORIES}
    for category, terms in QUERY_TERMS.items():
        overlap = len(tokens & terms)
        if overlap:
            targets[category] = min(1.0, overlap / 3.0)
    lowered = query.lower()
    if "what is" in lowered or "what are" in lowered:
        targets["definition_or_taxonomy"] = max(targets["definition_or_taxonomy"], 0.6)
    if "public opinion" in lowered:
        targets["public_expression"] = max(targets["public_expression"], 0.9)
    if "case law" in lowered or "legal authority" in lowered:
        targets["legal_authority"] = max(targets["legal_authority"], 0.9)
    if "risk factor" in lowered or "forward-looking" in lowered:
        targets["risk_disclosure"] = max(targets["risk_disclosure"], 0.9)
    if "how many" in lowered or "how much" in lowered:
        targets["quantitative_result"] = max(targets["quantitative_result"], 0.7)
    if not any(targets.values()):
        targets["generic_claim"] = 0.4
    return normalize_vector(targets)


def document_syntax_vector(document: RetrievalDocument) -> dict[str, float]:
    tokens = set(tokenize(" ".join([document.title, document.section, document.text])))
    vector = {category: 0.0 for category in SYNTAX_CATEGORIES}
    for category, terms in DOCUMENT_TERMS.items():
        vector[category] = min(1.0, len(tokens & terms) / 4.0)

    importance = document.chunk_importance
    vector["method_detail"] = max(vector["method_detail"], float(importance.get("method_signal", 0.0) or 0.0))
    vector["limitation_caveat"] = max(
        vector["limitation_caveat"],
        float(importance.get("limitation_signal", 0.0) or 0.0),
        float(importance.get("uncertainty_signal", 0.0) or 0.0) * 0.8,
    )
    vector["quantitative_result"] = max(
        vector["quantitative_result"],
        float(importance.get("evidence_specificity", 0.0) or 0.0),
        0.8 if QUANT_PATTERN.search(document.text) else 0.0,
    )
    vector["generic_claim"] = max(vector["generic_claim"], float(importance.get("claim_density", 0.0) or 0.0))

    if SPEAKER_PATTERN.search(document.text):
        vector["dialogue_or_transcript"] = max(vector["dialogue_or_transcript"], 0.85)
    if DEFINITION_PATTERN.search(document.text):
        vector["definition_or_taxonomy"] = max(vector["definition_or_taxonomy"], 0.85)

    work = document.work_signals
    for key in [
        str(work.get("document_type", "")),
        str(work.get("provenance", "")),
        str(work.get("evidence_family", "")),
        str(work.get("domain_profile", "")),
        str(work.get("domain", "")),
    ]:
        category = DOCUMENT_TYPE_CATEGORY.get(key) or EVIDENCE_FAMILY_CATEGORY.get(key)
        if category:
            vector[category] = max(vector[category], 0.85)

    if str(work.get("domain", "")).lower() in {"history", "anthropology", "cultural_studies"}:
        vector["narrative_context"] = max(vector["narrative_context"], 0.55)
    if str(work.get("document_type", "")).lower() == "regulatory_filing":
        vector["legal_authority"] = max(vector["legal_authority"], 0.45)

    return normalize_vector(vector)


def primary_syntax_category(vector: dict[str, float]) -> str:
    if not vector:
        return "generic_claim"
    category, value = max(vector.items(), key=lambda item: (item[1], item[0]))
    return category if value > 0 else "generic_claim"


def corpus_syntax_profile(documents: Sequence[RetrievalDocument]) -> SyntaxProfile:
    counts = {category: 0 for category in SYNTAX_CATEGORIES}
    vectors = [document_syntax_vector(document) for document in documents]
    for vector in vectors:
        for category, value in vector.items():
            if value >= 0.25:
                counts[category] += 1
    n = max(len(vectors), 1)
    raw_weights = {
        category: math.log((n + 1.0) / (counts[category] + 1.0)) + 1.0
        for category in SYNTAX_CATEGORIES
    }
    high = max(raw_weights.values()) if raw_weights else 1.0
    weights = {category: value / high for category, value in raw_weights.items()}
    return SyntaxProfile(category_counts=counts, category_weights=weights)


def syntax_relevance_score(query: str, document: RetrievalDocument, profile: SyntaxProfile | None = None) -> float:
    targets = query_syntax_targets(query)
    vector = document_syntax_vector(document)
    return syntax_alignment_score(targets, vector, profile)


def syntax_relevance_scores(query: str, documents: Sequence[RetrievalDocument]) -> list[float]:
    profile = corpus_syntax_profile(documents)
    targets = query_syntax_targets(query)
    return [syntax_alignment_score(targets, document_syntax_vector(document), profile) for document in documents]


def syntax_alignment_score(
    targets: dict[str, float],
    vector: dict[str, float],
    profile: SyntaxProfile | None = None,
) -> float:
    target_mass = sum(max(value, 0.0) for value in targets.values())
    if target_mass <= 0:
        return 0.0
    direct = sum(targets.get(category, 0.0) * vector.get(category, 0.0) for category in SYNTAX_CATEGORIES) / target_mass
    if not profile:
        return round(clamp(direct), 6)
    rarity = sum(
        targets.get(category, 0.0) * vector.get(category, 0.0) * profile.category_weights.get(category, 1.0)
        for category in SYNTAX_CATEGORIES
    ) / target_mass
    return round(clamp((0.75 * direct) + (0.25 * rarity)), 6)


def normalize_vector(values: dict[str, float]) -> dict[str, float]:
    return {category: round(clamp(float(values.get(category, 0.0) or 0.0)), 6) for category in SYNTAX_CATEGORIES}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
