from __future__ import annotations

import os
import socket
import time
import urllib.error
from dataclasses import dataclass
from typing import Any, Protocol

from canon.http_client import post_json
from canon.secrets import load_local_env


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class GenerationResult:
    provider: str
    model: str
    text: str
    raw: dict | None = None


class GenerationProviderError(RuntimeError):
    def __init__(
        self,
        failure_type: str,
        message: str,
        *,
        model: str | None = None,
        http_status: int | None = None,
        retryable: bool = False,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.model = model
        self.http_status = http_status
        self.retryable = retryable
        self.attempts = attempts


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


class OpenRouterGenerationProvider:
    provider = "openrouter"

    def __init__(
        self,
        model: str = "openai/gpt-4.1-mini",
        api_key: str | None = None,
        *,
        timeout_s: float | None = None,
        max_attempts: int | None = None,
    ) -> None:
        load_local_env()
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise GenerationProviderError(
                "missing_key",
                "OPENROUTER_API_KEY is required for OpenRouter generation.",
                model=self.model,
                retryable=False,
            )
        self.timeout_s = timeout_s or default_openrouter_timeout_s(model)
        self.max_attempts = max_attempts or int(os.getenv("CANON_GENERATION_MAX_ATTEMPTS", "3"))
        self.max_attempts = max(1, self.max_attempts)

    def generate(self, prompt: str) -> GenerationResult:
        payload = self.post_with_retries(
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
            }
        )
        text = extract_chat_completion_text(payload)
        return GenerationResult(
            provider=self.provider,
            model=payload.get("model") or self.model,
            text=text,
            raw={"id": payload.get("id"), "model": payload.get("model"), "usage": payload.get("usage")},
        )

    def post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: GenerationProviderError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return post_json(
                    OPENROUTER_CHAT_COMPLETIONS_URL,
                    payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "CANON local workbench",
                    },
                    timeout=self.timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 - provider boundary converts opaque failures.
                last_error = classify_generation_exception(exc, model=self.model, attempts=attempt)
                if not last_error.retryable or attempt >= self.max_attempts:
                    raise last_error from exc
                time.sleep(retry_delay_s(attempt))
        if last_error is not None:
            raise last_error
        raise GenerationProviderError("provider_error", "OpenRouter generation failed.", model=self.model)


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


def default_openrouter_timeout_s(model: str) -> float:
    normalized = model.lower()
    if "kimi-k3" in normalized:
        return 90.0
    if "thinking" in normalized or "kimi-k2" in normalized:
        return 75.0
    if "gpt-4.1-mini" in normalized or "gpt-4o-mini" in normalized:
        return 45.0
    return 60.0


def retry_delay_s(attempt: int) -> float:
    return min(4.0, 0.5 * (2 ** max(0, attempt - 1)))


def classify_generation_exception(exc: Exception, *, model: str | None = None, attempts: int = 1) -> GenerationProviderError:
    if isinstance(exc, GenerationProviderError):
        return GenerationProviderError(
            exc.failure_type,
            str(exc),
            model=exc.model or model,
            http_status=exc.http_status,
            retryable=exc.retryable,
            attempts=attempts,
        )

    http_status = http_status_from_exception(exc)
    detail = compact_error_detail(exc)
    if http_status is not None:
        failure_type = failure_type_from_http_status(http_status, detail)
        return GenerationProviderError(
            failure_type,
            f"OpenRouter generation failed ({failure_type}; HTTP {http_status}). {detail}",
            model=model,
            http_status=http_status,
            retryable=http_status in RETRYABLE_HTTP_STATUSES,
            attempts=attempts,
        )

    class_name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    if isinstance(exc, (TimeoutError, socket.timeout)) or "timeout" in class_name or "timed out" in text:
        return GenerationProviderError(
            "timeout",
            f"OpenRouter generation timed out. {detail}",
            model=model,
            retryable=True,
            attempts=attempts,
        )
    if isinstance(exc, urllib.error.URLError) or "connect" in class_name or "network" in class_name:
        return GenerationProviderError(
            "network_error",
            f"OpenRouter generation could not reach the provider. {detail}",
            model=model,
            retryable=True,
            attempts=attempts,
        )
    if isinstance(exc, ValueError):
        return GenerationProviderError(
            "malformed_response",
            f"OpenRouter generation returned malformed JSON. {detail}",
            model=model,
            retryable=True,
            attempts=attempts,
        )
    return GenerationProviderError(
        "provider_error",
        f"OpenRouter generation failed. {detail}",
        model=model,
        retryable=False,
        attempts=attempts,
    )


def http_status_from_exception(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    if isinstance(exc, urllib.error.HTTPError):
        return int(exc.code)
    return None


def failure_type_from_http_status(status: int, detail: str) -> str:
    lowered = detail.lower()
    if status in {401, 403}:
        return "auth_failed"
    if status == 402:
        return "insufficient_credits"
    if status == 404 or (status == 400 and "model" in lowered):
        return "invalid_model"
    if status == 408:
        return "timeout"
    if status == 429:
        return "rate_limited"
    if status in {500, 502, 503, 504}:
        return "provider_5xx"
    if status == 400:
        return "bad_request"
    return f"http_{status}"


def compact_error_detail(exc: Exception, limit: int = 320) -> str:
    detail = ""
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None)
    if response_text:
        detail = str(response_text)
    elif isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - best-effort diagnostic body.
            detail = str(exc)
    else:
        detail = str(exc)
    secret = os.getenv("OPENROUTER_API_KEY")
    if secret:
        detail = detail.replace(secret, "[redacted]")
    detail = " ".join(detail.split())
    return detail[:limit]


def get_generation_provider(name: str, model: str | None = None) -> GenerationProvider:
    normalized = name.lower()
    if normalized in {"template", "local"}:
        return TemplateGenerationProvider()
    if normalized == "openai":
        return OpenRouterGenerationProvider(model=openrouter_model_id(model or "gpt-4.1-mini"))
    if normalized == "openrouter":
        return OpenRouterGenerationProvider(model=model or "openai/gpt-4.1-mini")
    raise ValueError(f"Unknown generation provider: {name}")


def openrouter_model_id(model: str) -> str:
    return model if "/" in model else f"openai/{model}"
