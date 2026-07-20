from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from canon.secrets import load_local_env


FRAME_FIELDS = {
    "topic",
    "question_type",
    "subdomains",
    "regions",
    "languages",
    "disciplines",
    "methods",
    "representation_goals",
    "exclusion_criteria",
    "review_risks",
}


def enrich_frame(
    question: str,
    deterministic_frame: dict[str, Any],
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    if not provider or provider in {"none", "deterministic"}:
        return enrichment_result("deterministic_only", deterministic_frame, {}, "No model enrichment requested.")
    if provider != "openrouter":
        return enrichment_result("unavailable", deterministic_frame, {}, f"Unsupported frame enrichment provider: {provider}")
    try:
        suggestion = openrouter_frame_suggestion(question, deterministic_frame, model or "openai/gpt-4.1-mini")
    except Exception as exc:  # noqa: BLE001
        return enrichment_result("unavailable", deterministic_frame, {}, str(exc))
    merged = merge_frames(deterministic_frame, suggestion)
    return enrichment_result("model_suggested", merged, suggestion, "Model suggestions merged under deterministic schema.")


def openrouter_frame_suggestion(question: str, deterministic_frame: dict[str, Any], model: str) -> dict[str, Any]:
    load_local_env()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter frame enrichment.")
    prompt = {
        "instruction": (
            "Return only JSON for a research frame. Keep values concise. "
            "Do not invent proper nouns unless implied by the question."
        ),
        "question": question,
        "current_frame": deterministic_frame,
        "allowed_fields": sorted(FRAME_FIELDS),
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "You produce strict JSON research-frame suggestions."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Frame enrichment response must be a JSON object.")
    return sanitize_frame(parsed)


def merge_frames(base: dict[str, Any], suggestion: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in sanitize_frame(suggestion).items():
        if key not in FRAME_FIELDS:
            continue
        if isinstance(value, list):
            merged[key] = merge_lists(merged.get(key), value)
        elif value and not merged.get(key):
            merged[key] = value
    return merged


def sanitize_frame(frame: dict[str, Any]) -> dict[str, Any]:
    sanitized = {}
    for key, value in frame.items():
        if key not in FRAME_FIELDS:
            continue
        if key in {
            "subdomains",
            "regions",
            "languages",
            "disciplines",
            "methods",
            "representation_goals",
            "exclusion_criteria",
            "review_risks",
        }:
            sanitized[key] = string_list(value)
        elif isinstance(value, str):
            sanitized[key] = value.strip()
    return sanitized


def merge_lists(left: Any, right: list[str]) -> list[str]:
    values = string_list(left) + string_list(right)
    seen = set()
    merged = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return merged


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [piece.strip() for piece in value.split(",") if piece.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def enrichment_result(status: str, frame: dict[str, Any], suggestion: dict[str, Any], note: str) -> dict[str, Any]:
    return {
        "status": status,
        "research_frame": frame,
        "model_suggestion": suggestion,
        "note": note,
        "human_review_required": True,
    }
