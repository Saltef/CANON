from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from canon.retrieval.semantic import HashedSemanticEncoder


@dataclass(frozen=True)
class EmbeddingResult:
    provider: str
    model: str
    dimensions: int
    vector: list[float]


class EmbeddingProvider(Protocol):
    provider: str
    model: str

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        ...


class HashedEmbeddingProvider:
    provider = "local"
    model = "hashed-semantic-v1"

    def __init__(self, dimensions: int = 512) -> None:
        self.encoder = HashedSemanticEncoder(dimensions=dimensions, model_id=self.model)

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        results = []
        for text in texts:
            embedding = self.encoder.encode(text)
            vector = [0.0] * embedding.dimensions
            for index, value in embedding.values.items():
                vector[index] = value
            results.append(
                EmbeddingResult(
                    provider=self.provider,
                    model=self.model,
                    dimensions=embedding.dimensions,
                    vector=vector,
                )
            )
        return results


class OpenAIEmbeddingProvider:
    provider = "openai"

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return [
            EmbeddingResult(
                provider=self.provider,
                model=self.model,
                dimensions=len(row["embedding"]),
                vector=[float(value) for value in row["embedding"]],
            )
            for row in sorted(payload["data"], key=lambda item: item["index"])
        ]


class CohereEmbeddingProvider:
    provider = "cohere"

    def __init__(
        self,
        model: str = "embed-v4.0",
        api_key: str | None = None,
        input_type: str = "search_document",
    ) -> None:
        self.model = model
        self.input_type = input_type
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise RuntimeError("COHERE_API_KEY is required for Cohere embeddings.")

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        request = urllib.request.Request(
            "https://api.cohere.com/v2/embed",
            data=json.dumps(
                {
                    "model": self.model,
                    "texts": texts,
                    "input_type": self.input_type,
                    "embedding_types": ["float"],
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        vectors = payload["embeddings"]["float"]
        return [
            EmbeddingResult(
                provider=self.provider,
                model=self.model,
                dimensions=len(vector),
                vector=[float(value) for value in vector],
            )
            for vector in vectors
        ]


def get_embedding_provider(name: str, model: str | None = None) -> EmbeddingProvider:
    normalized = name.lower()
    if normalized in {"local", "hashed"}:
        return HashedEmbeddingProvider()
    if normalized == "openai":
        return OpenAIEmbeddingProvider(model=model or "text-embedding-3-small")
    if normalized == "cohere":
        return CohereEmbeddingProvider(model=model or "embed-v4.0")
    raise ValueError(f"Unknown embedding provider: {name}")
