from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from canon.http_client import post_json
from canon.secrets import load_local_env


@dataclass(frozen=True)
class GenerationResult:
    provider: str
    model: str
    text: str
    raw: dict | None = None


class GenerationProvider(Protocol):
    provider: str
    model: str

    def generate(self, prompt: str) -> GenerationResult:
        ...


class TemplateGenerationProvider:
    provider = "template"
    model = "grounded-template-v1"

    def generate(self, prompt: str) -> GenerationResult:
        return GenerationResult(provider=self.provider, model=self.model, text=prompt)


class OpenAIGenerationProvider:
    provider = "openai"

    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None) -> None:
        load_local_env()
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI generation.")

    def generate(self, prompt: str) -> GenerationResult:
        payload = post_json(
            "https://api.openai.com/v1/responses",
            {
                "model": self.model,
                "input": prompt,
                "temperature": 0,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )
        text = payload.get("output_text") or extract_response_text(payload)
        return GenerationResult(
            provider=self.provider,
            model=self.model,
            text=text,
            raw={"id": payload.get("id"), "model": payload.get("model")},
        )


class OpenRouterGenerationProvider:
    provider = "openrouter"

    def __init__(self, model: str = "openai/gpt-4.1-mini", api_key: str | None = None) -> None:
        load_local_env()
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter generation.")

    def generate(self, prompt: str) -> GenerationResult:
        payload = post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You write concise evidence-grounded research notes. "
                            "Use only supplied evidence and cite every factual sentence."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 700,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "CANON local workbench",
            },
            timeout=120,
        )
        text = extract_chat_completion_text(payload)
        return GenerationResult(
            provider=self.provider,
            model=payload.get("model") or self.model,
            text=text,
            raw={"id": payload.get("id"), "model": payload.get("model")},
        )


def extract_response_text(payload: dict) -> str:
    parts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                parts.append(content.get("text", ""))
    return "\n".join(part for part in parts if part).strip()


def extract_chat_completion_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or part) for part in content).strip()
    return str(content).strip()


def get_generation_provider(name: str, model: str | None = None) -> GenerationProvider:
    normalized = name.lower()
    if normalized in {"template", "local"}:
        return TemplateGenerationProvider()
    if normalized == "openai":
        return OpenAIGenerationProvider(model=model or "gpt-4.1-mini")
    if normalized == "openrouter":
        return OpenRouterGenerationProvider(model=model or "openai/gpt-4.1-mini")
    raise ValueError(f"Unknown generation provider: {name}")
