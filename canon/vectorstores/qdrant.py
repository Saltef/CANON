from __future__ import annotations

from typing import Any

from canon.vectorstores.base import VectorPoint, VectorSearchResult, VectorStoreError


class QdrantVectorStore:
    backend = "qdrant"

    def __init__(self, url: str | None = None, api_key: str | None = None, client: Any | None = None) -> None:
        if client is not None:
            self.client = client
            return
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise VectorStoreError(
                "qdrant-client is required for the Qdrant vector backend. "
                "Install the vectorstores extra or use vector_backend=memory for tests."
            ) from exc
        self.client = QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, collection: str, vector_size: int, distance: str = "Cosine") -> None:
        if self._collection_exists(collection):
            return
        try:
            from qdrant_client.http import models
        except ImportError as exc:
            raise VectorStoreError("qdrant-client models are required to create collections.") from exc
        self.client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=getattr(models.Distance, distance.upper(), models.Distance.COSINE),
            ),
        )

    def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        if not points:
            return
        try:
            from qdrant_client.http import models
        except ImportError as exc:
            raise VectorStoreError("qdrant-client models are required to upsert points.") from exc
        self.client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in points
            ],
        )

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        filters = filters or {}
        self._ensure_payload_indexes(collection, filters)
        query_filter = self._filter(filters)
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            points = getattr(result, "points", result)
        else:
            points = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        return [
            VectorSearchResult(
                id=str(getattr(point, "id", "")),
                score=float(getattr(point, "score", 0.0) or 0.0),
                payload=dict(getattr(point, "payload", {}) or {}),
            )
            for point in points
        ]

    def delete_points(self, collection: str, point_ids: list[str]) -> None:
        if not point_ids:
            return
        try:
            from qdrant_client.http import models
        except ImportError as exc:
            raise VectorStoreError("qdrant-client models are required to delete points.") from exc
        self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=point_ids),
        )

    def _collection_exists(self, collection: str) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(collection_name=collection))
        try:
            self.client.get_collection(collection_name=collection)
            return True
        except Exception:  # noqa: BLE001 - qdrant clients expose different exception classes.
            return False

    def _filter(self, filters: dict[str, Any]) -> Any | None:
        if not filters:
            return None
        try:
            from qdrant_client.http import models
        except ImportError as exc:
            raise VectorStoreError("qdrant-client models are required to build filters.") from exc
        return models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in filters.items()
            ]
        )

    def _ensure_payload_indexes(self, collection: str, filters: dict[str, Any]) -> None:
        if not filters or not hasattr(self.client, "create_payload_index"):
            return
        try:
            from qdrant_client.http import models
        except ImportError as exc:
            raise VectorStoreError("qdrant-client models are required to create payload indexes.") from exc
        for key, value in filters.items():
            try:
                self.client.create_payload_index(
                    collection_name=collection,
                    field_name=key,
                    field_schema=payload_schema_type(models, value),
                )
            except Exception as exc:  # noqa: BLE001 - qdrant versions return varied exists errors.
                message = str(exc).lower()
                if "already exists" in message or "already has" in message:
                    continue
                if "conflict" in message or "same params" in message:
                    continue
                raise


def payload_schema_type(models: Any, value: Any) -> Any:
    if isinstance(value, bool):
        return models.PayloadSchemaType.BOOL
    if isinstance(value, int):
        return models.PayloadSchemaType.INTEGER
    if isinstance(value, float):
        return models.PayloadSchemaType.FLOAT
    return models.PayloadSchemaType.KEYWORD
