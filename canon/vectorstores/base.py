from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from canon.embeddings.store import artifact_key
from canon.secrets import load_local_env


class VectorStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore(Protocol):
    backend: str

    def ensure_collection(self, collection: str, vector_size: int, distance: str = "Cosine") -> None:
        ...

    def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        ...

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        ...

    def delete_points(self, collection: str, point_ids: list[str]) -> None:
        ...


class MemoryVectorStore:
    backend = "memory"

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}

    def ensure_collection(self, collection: str, vector_size: int, distance: str = "Cosine") -> None:
        existing = self.collections.get(collection)
        if existing and existing["vector_size"] != vector_size:
            raise VectorStoreError(
                f"Collection {collection} has vector size {existing['vector_size']}, expected {vector_size}."
            )
        self.collections.setdefault(
            collection,
            {"vector_size": vector_size, "distance": distance, "points": {}},
        )

    def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        if not points:
            return
        if collection not in self.collections:
            self.ensure_collection(collection, len(points[0].vector))
        store = self.collections[collection]["points"]
        for point in points:
            store[point.id] = point

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        points = (self.collections.get(collection) or {}).get("points") or {}
        scored = []
        for point in points.values():
            if not payload_matches(point.payload, filters or {}):
                continue
            scored.append(
                VectorSearchResult(
                    id=point.id,
                    score=dense_cosine(query_vector, point.vector),
                    payload=dict(point.payload),
                )
            )
        return sorted(scored, key=lambda row: (row.score, row.id), reverse=True)[:top_k]

    def delete_points(self, collection: str, point_ids: list[str]) -> None:
        points = (self.collections.get(collection) or {}).get("points") or {}
        for point_id in point_ids:
            points.pop(point_id, None)


def get_vector_store(
    backend: str = "qdrant",
    *,
    url: str | None = None,
    api_key: str | None = None,
    client: Any | None = None,
) -> VectorStore:
    normalized = backend.lower().replace("-", "_")
    if normalized in {"memory", "local", "in_memory"}:
        return MemoryVectorStore()
    if normalized == "qdrant":
        from canon.vectorstores.qdrant import QdrantVectorStore

        load_local_env()
        resolved_url = url or os.getenv("QDRANT_URL")
        resolved_api_key = api_key or os.getenv("QDRANT_API_KEY")
        if client is None and not resolved_url:
            raise VectorStoreError("QDRANT_URL is required for the Qdrant vector backend.")
        return QdrantVectorStore(url=resolved_url, api_key=resolved_api_key, client=client)
    raise ValueError(f"Unknown vector backend: {backend}")


def vector_collection_name(mode: str, embedding_provider: str, embedding_model: str | None = None) -> str:
    suffix = artifact_key(embedding_provider, embedding_model)
    return safe_collection_name(f"canon_{mode}_{suffix}")


def point_id_for_chunk(collection: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"canon-vector-point:{collection}:{chunk_id}"))


def payload_matches(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        if payload.get(key) != value:
            return False
    return True


def dense_cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def safe_collection_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"_", "-"} else "-" for character in value)
    cleaned = cleaned.strip("-_").lower()
    return cleaned or "canon"
