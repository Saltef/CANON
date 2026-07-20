from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from canon.secrets import load_local_env
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
        load_local_env()
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


class OpenRouterEmbeddingProvider:
    provider = "openrouter"

    def __init__(self, model: str = "openai/text-embedding-3-small", api_key: str | None = None) -> None:
        load_local_env()
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter embeddings.")

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/embeddings",
            data=json.dumps({"model": self.model, "input": texts, "encoding_format": "float"}).encode("utf-8"),
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
                model=payload.get("model") or self.model,
                dimensions=len(row["embedding"]),
                vector=[float(value) for value in row["embedding"]],
            )
            for row in sorted(payload["data"], key=lambda item: item["index"])
        ]

    def embed_queries(self, texts: list[str]) -> list[EmbeddingResult]:
        return self.embed([prepare_openrouter_query(self.model, text) for text in texts])


class CohereEmbeddingProvider:
    provider = "cohere"

    def __init__(
        self,
        model: str = "embed-v4.0",
        api_key: str | None = None,
        input_type: str = "search_document",
    ) -> None:
        load_local_env()
        self.model = model
        self.input_type = input_type
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise RuntimeError("COHERE_API_KEY is required for Cohere embeddings.")

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        return self.embed_with_input_type(texts, self.input_type)

    def embed_with_input_type(self, texts: list[str], input_type: str) -> list[EmbeddingResult]:
        request = urllib.request.Request(
            "https://api.cohere.com/v2/embed",
            data=json.dumps(
                {
                    "model": self.model,
                    "texts": texts,
                    "input_type": input_type,
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
    if normalized == "openrouter":
        return OpenRouterEmbeddingProvider(model=model or "openai/text-embedding-3-small")
    if normalized == "cohere":
        return CohereEmbeddingProvider(model=model or "embed-v4.0")
    raise ValueError(f"Unknown embedding provider: {name}")


def embed_documents(provider: EmbeddingProvider, texts: list[str]) -> list[EmbeddingResult]:
    return provider.embed(texts)


def embed_queries(provider: EmbeddingProvider, texts: list[str]) -> list[EmbeddingResult]:
    if isinstance(provider, CohereEmbeddingProvider):
        return provider.embed_with_input_type(texts, "search_query")
    if isinstance(provider, OpenRouterEmbeddingProvider):
        return provider.embed_queries(texts)
    return provider.embed(texts)


def prepare_openrouter_query(model: str, query: str) -> str:
    if model.startswith("qwen/qwen3-embedding"):
        task = "Given a scientific claim or research question, retrieve relevant passages that support, refute, or answer it"
        return f"Instruct: {task}\nQuery: {query}"
    return query


def embedding_preparation(provider: EmbeddingProvider) -> dict[str, str]:
    if isinstance(provider, CohereEmbeddingProvider):
        return {
            "document_input": "Cohere input_type=search_document",
            "query_input": "Cohere input_type=search_query",
        }
    if isinstance(provider, OpenRouterEmbeddingProvider) and provider.model.startswith("qwen/qwen3-embedding"):
        return {
            "document_input": "raw document text",
            "query_input": "Qwen3 retrieval instruction prefix plus raw query",
        }
    if isinstance(provider, OpenRouterEmbeddingProvider) and provider.model == "baai/bge-m3":
        return {
            "document_input": "raw document text",
            "query_input": "raw query text; hybrid BM25+dense retrieval remains in the first-stage pool",
        }
    return {"document_input": "raw document text", "query_input": "raw query text"}
