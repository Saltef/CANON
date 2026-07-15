from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol

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
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(
                {
                    "model": self.model,
                    "input": prompt,
                    "temperature": 0,
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = payload.get("output_text") or extract_response_text(payload)
        return GenerationResult(
            provider=self.provider,
            model=self.model,
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


def get_generation_provider(name: str, model: str | None = None) -> GenerationProvider:
    normalized = name.lower()
    if normalized in {"template", "local"}:
        return TemplateGenerationProvider()
    if normalized == "openai":
        return OpenAIGenerationProvider(model=model or "gpt-4.1-mini")
    raise ValueError(f"Unknown generation provider: {name}")
