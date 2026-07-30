from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from canon.retrieval.query_diagnostics import content_terms
from canon.secrets import load_local_env


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


class RerankProvider(Protocol):
    provider: str
    model: str

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        ...


class HeuristicRerankProvider:
    provider = "heuristic"
    model = "lexical-overlap-rerank-v1"

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        query_terms = set(content_terms(query))
        rows = []
        for index, document in enumerate(documents):
            terms = set(content_terms(document))
            score = safe_ratio(len(query_terms & terms), len(query_terms))
            rows.append(RerankResult(index=index, score=score))
        return sorted(rows, key=lambda row: (row.score, -row.index), reverse=True)[:top_n]


class CohereRerankProvider:
    provider = "cohere"

    def __init__(self, model: str = "rerank-v4.0-pro", api_key: str | None = None) -> None:
        load_local_env()
        self.model = model
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise RuntimeError("COHERE_API_KEY is required for Cohere rerank.")

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        request = urllib.request.Request(
            "https://api.cohere.com/v2/rerank",
            data=json.dumps(
                {
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
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
        return [
            RerankResult(index=int(row["index"]), score=float(row["relevance_score"]))
            for row in payload.get("results", [])
        ]


class OpenRouterRerankProvider:
    provider = "openrouter"

    def __init__(self, model: str = "cohere/rerank-v3.5", api_key: str | None = None) -> None:
        load_local_env()
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter rerank.")

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/rerank",
            data=json.dumps(
                {
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
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
        return [
            RerankResult(index=int(row["index"]), score=float(row["relevance_score"]))
            for row in payload.get("results", [])
        ]


def get_rerank_provider(name: str, model: str | None = None) -> RerankProvider:
    normalized = name.lower()
    if normalized in {"heuristic", "local"}:
        return HeuristicRerankProvider()
    if normalized == "cohere":
        return CohereRerankProvider(model=model or "rerank-v4.0-pro")
    if normalized == "openrouter":
        return OpenRouterRerankProvider(model=model or "cohere/rerank-v3.5")
    raise ValueError(f"Unknown rerank provider: {name}")


def safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
