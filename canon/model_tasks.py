from __future__ import annotations

import json
import os
from typing import Any

from canon.http_client import post_json
from canon.retrieval.query_diagnostics import content_terms, term_matches_tokens
from canon.retrieval.tokenize import tokenize
from canon.secrets import load_local_env


MODEL_TASK_SCHEMA_VERSION = "canon_model_task_review_v1"
DEFAULT_MODEL_TASK_PROVIDER = "openrouter"
DEFAULT_MODEL_TASK_MODEL = "openai/gpt-4.1-mini"
ALLOWED_STANCES = {"supports", "contradicts", "qualifies", "neutral", "insufficient"}
ALLOWED_AXES = {
    "same_question_different_answer",
    "different_question_disguised_as_conflict",
    "method_difference",
    "population_difference",
    "time_period_difference",
    "measurement_difference",
    "evidence_gap",
    "not_enough_evidence",
}


def run_evidence_model_review(
    *,
    query: str,
    evidence: list[dict[str, Any]],
    provider: str = DEFAULT_MODEL_TASK_PROVIDER,
    model: str | None = None,
    allow_external_data: bool = False,
    max_evidence: int = 8,
) -> dict[str, Any]:
    normalized = provider.lower()
    model = model or default_model_task_model(normalized)
    if normalized in {"deterministic", "local", "template"}:
        return deterministic_model_review(query, evidence[:max_evidence])
    if not allow_external_data:
        raise RuntimeError(
            "Model review would send evidence snippets to a hosted provider. "
            "Set allow_external_model_review=true only for approved corpora."
        )
    if normalized != "openrouter":
        raise ValueError("Model review currently supports openrouter or deterministic providers.")
    validate_openrouter_model(model)
    payload = call_openrouter_model_review(model, model_review_prompt(query, evidence[:max_evidence]))
    return sanitize_model_review_payload(
        payload,
        query=query,
        evidence=evidence[:max_evidence],
        provider=normalized,
        model=model,
    )


def deterministic_model_review(query: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    focus_terms = content_terms(query)
    rows = []
    for item in evidence:
        text = evidence_text(item)
        tokens = set(tokenize(text))
        overlap = [term for term in focus_terms if term_matches_tokens(term, tokens)]
        stance = "qualifies" if overlap else "insufficient"
        rows.append(
            {
                "evidence_id": str(item.get("evidence_id") or ""),
                "stance": stance,
                "claim": compact_text(best_sentence(text, overlap) or text, 220),
                "confidence": round(min(0.65, 0.2 + 0.12 * len(overlap)), 6),
                "excerpt": compact_text(best_sentence(text, overlap) or text, 280),
                "rationale": "Deterministic overlap fallback; use hosted model review for real stance analysis.",
            }
        )
    return {
        "report_id": "evidence_model_review_v1",
        "schema_version": MODEL_TASK_SCHEMA_VERSION,
        "status": "deterministic_fallback",
        "provider": "deterministic",
        "model": "lexical-model-review-v1",
        "query": query,
        "stance_assessments": rows,
        "extracted_dimensions": [],
        "disagreement_diagnosis": {
            "axis": "not_enough_evidence",
            "confidence": 0.0,
            "summary": "Hosted model review was not run.",
            "evidence_ids": [row["evidence_id"] for row in rows if row["evidence_id"]],
        },
        "draft_plan": {
            "status": "not_generated",
            "safe_to_draft": False,
            "missing_evidence": ["Run hosted model review or human review for stance/extraction."],
        },
        "human_review_required": True,
        "boundary": "Deterministic model review is a development fallback, not a stance-quality claim.",
    }


def call_openrouter_model_review(model: str, prompt: dict[str, Any]) -> dict[str, Any]:
    load_local_env()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for hosted model review.")
    payload = post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful evidence analyst. Return strict JSON only. "
                        "Use only supplied evidence IDs and excerpts."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": 1400,
            "response_format": {"type": "json_object"},
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "CANON evidence workbench",
        },
        timeout=180,
    )
    content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "{}"
    return json.loads(content)


def model_review_prompt(query: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": "Classify evidence stance and diagnose disagreement for a research workbench.",
        "query": query,
        "rules": [
            "Return JSON only.",
            "Use only evidence_id values supplied below.",
            "Every stance assessment must cite one evidence_id and a short excerpt.",
            "Classify stance relative to the user query, not whether the source is generally important.",
            "Use insufficient when the evidence does not visibly answer the query.",
            "Do not write final conclusions; this is review assistance.",
        ],
        "stance_values": sorted(ALLOWED_STANCES),
        "diagnosis_axes": sorted(ALLOWED_AXES),
        "output_schema": {
            "stance_assessments": [
                {
                    "evidence_id": "C1",
                    "stance": "supports | contradicts | qualifies | neutral | insufficient",
                    "claim": "atomic claim supported by this evidence",
                    "confidence": "0..1",
                    "excerpt": "short quoted or near-quoted span",
                    "rationale": "one sentence",
                }
            ],
            "extracted_dimensions": [
                {
                    "evidence_id": "C1",
                    "dimension": "method | population | geography | time_period | measurement | limitation | number",
                    "value": "extracted value",
                    "excerpt": "span supporting the value",
                }
            ],
            "disagreement_diagnosis": {
                "axis": "one diagnosis_axes value",
                "confidence": "0..1",
                "summary": "short diagnosis",
                "evidence_ids": ["C1"],
            },
            "draft_plan": {
                "safe_to_draft": "boolean",
                "missing_evidence": ["what is missing"],
                "suggested_next_queries": ["query refinements"],
            },
        },
        "evidence": [
            {
                "evidence_id": str(item.get("evidence_id") or ""),
                "title": item.get("title"),
                "source_name": item.get("source_name"),
                "source_type": item.get("source_type"),
                "text": evidence_text(item)[:1200],
            }
            for item in evidence
            if item.get("evidence_id")
        ],
    }


def sanitize_model_review_payload(
    payload: dict[str, Any],
    *,
    query: str,
    evidence: list[dict[str, Any]],
    provider: str,
    model: str,
) -> dict[str, Any]:
    evidence_by_id = {str(item.get("evidence_id")): item for item in evidence if item.get("evidence_id")}
    stances = [
        row
        for row in (sanitize_stance(row, evidence_by_id) for row in payload.get("stance_assessments") or [])
        if row
    ]
    dimensions = [
        row
        for row in (sanitize_dimension(row, evidence_by_id) for row in payload.get("extracted_dimensions") or [])
        if row
    ]
    diagnosis = sanitize_diagnosis(payload.get("disagreement_diagnosis") or {}, evidence_by_id)
    draft_plan = sanitize_draft_plan(payload.get("draft_plan") or {})
    return {
        "report_id": "evidence_model_review_v1",
        "schema_version": MODEL_TASK_SCHEMA_VERSION,
        "status": "model_review_ready" if stances or dimensions else "empty_model_review",
        "provider": provider,
        "model": model,
        "query": query,
        "stance_assessments": stances,
        "extracted_dimensions": dimensions,
        "disagreement_diagnosis": diagnosis,
        "draft_plan": draft_plan,
        "human_review_required": True,
        "boundary": "Model review is typed triage over retrieved evidence; it is not a ground-truth label.",
    }


def sanitize_stance(row: Any, evidence_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    evidence_id = str(row.get("evidence_id") or "")
    if evidence_id not in evidence_by_id:
        return None
    stance = str(row.get("stance") or "").lower()
    if stance not in ALLOWED_STANCES:
        stance = "insufficient"
    return {
        "evidence_id": evidence_id,
        "stance": stance,
        "claim": compact_text(row.get("claim"), 260),
        "confidence": clamped_float(row.get("confidence")),
        "excerpt": compact_text(row.get("excerpt"), 360),
        "rationale": compact_text(row.get("rationale"), 260),
    }


def sanitize_dimension(row: Any, evidence_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    evidence_id = str(row.get("evidence_id") or "")
    if evidence_id not in evidence_by_id:
        return None
    return {
        "evidence_id": evidence_id,
        "dimension": compact_text(row.get("dimension"), 80).lower(),
        "value": compact_text(row.get("value"), 220),
        "excerpt": compact_text(row.get("excerpt"), 300),
    }


def sanitize_diagnosis(row: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    axis = str(row.get("axis") or "").lower()
    if axis not in ALLOWED_AXES:
        axis = "not_enough_evidence"
    evidence_ids = [
        str(evidence_id)
        for evidence_id in row.get("evidence_ids") or []
        if str(evidence_id) in evidence_by_id
    ]
    return {
        "axis": axis,
        "confidence": clamped_float(row.get("confidence")),
        "summary": compact_text(row.get("summary"), 360),
        "evidence_ids": evidence_ids,
    }


def sanitize_draft_plan(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "safe_to_draft": bool(row.get("safe_to_draft")),
        "missing_evidence": [compact_text(item, 160) for item in row.get("missing_evidence") or [] if compact_text(item, 160)],
        "suggested_next_queries": [
            compact_text(item, 180) for item in row.get("suggested_next_queries") or [] if compact_text(item, 180)
        ],
    }


def default_model_task_model(provider: str) -> str:
    if provider == "openrouter":
        return DEFAULT_MODEL_TASK_MODEL
    return "lexical-model-review-v1"


def validate_openrouter_model(model: str) -> None:
    normalized = model.lower()
    if normalized.startswith("cohere/"):
        raise ValueError("Use Cohere for reranking, not OpenRouter model review.")
    if "gemini" in normalized or normalized.startswith("google/"):
        raise ValueError("Gemini models are disabled for this CANON configuration.")


def evidence_text(item: dict[str, Any]) -> str:
    return compact_text(item.get("text") or item.get("preview") or item.get("claim") or "", 1500)


def best_sentence(text: str, terms: list[str]) -> str:
    sentences = [piece.strip() for piece in text.replace("\n", " ").split(".") if piece.strip()]
    if not sentences:
        return text
    term_set = set(terms)
    return max(sentences, key=lambda sentence: len(term_set & set(tokenize(sentence))))


def compact_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def clamped_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return round(max(0.0, min(1.0, number)), 6)
