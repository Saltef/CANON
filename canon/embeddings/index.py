from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.embeddings.providers import EmbeddingProvider, get_embedding_provider
from canon.embeddings.store import artifact_key, write_json
from canon.retrieval.clusters import load_cluster_assignments
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus
from canon.vectorstores import (
    VectorPoint,
    VectorStore,
    get_vector_store,
    point_id_for_chunk,
    vector_collection_name,
)


VECTOR_INDEX_SCHEMA_VERSION = "canon_vector_index_v1"


def build_vector_index(
    mode: str,
    embedding_provider_name: str = "openrouter",
    embedding_model: str | None = None,
    vector_backend: str = "qdrant",
    collection: str | None = None,
    batch_size: int = 32,
    delete_stale: bool = True,
    vector_store: VectorStore | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    documents = load_processed_corpus(
        settings.data_dir,
        mode,
        cluster_assignments=load_cluster_assignments(settings.data_dir, mode),
    )
    provider = get_embedding_provider(embedding_provider_name, embedding_model)
    collection = collection or vector_collection_name(mode, provider.provider, provider.model)
    store = vector_store or get_vector_store(vector_backend)
    report = index_documents(
        documents=documents,
        mode=mode,
        provider=provider,
        vector_store=store,
        vector_backend=vector_backend,
        collection=collection,
        batch_size=batch_size,
        previous_manifest=load_manifest(manifest_path(settings.data_dir, mode, vector_backend, provider.provider, provider.model)),
        delete_stale=delete_stale,
    )
    write_manifest(
        manifest_path(settings.data_dir, mode, vector_backend, provider.provider, provider.model),
        report["manifest"],
    )
    write_json(
        settings.reports_dir / f"vector_index_{mode}_{safe_slug(vector_backend)}_{artifact_key(provider.provider, provider.model)}.json",
        public_index_report(report),
    )
    return public_index_report(report)


def index_documents(
    *,
    documents: list[RetrievalDocument],
    mode: str,
    provider: EmbeddingProvider,
    vector_store: VectorStore,
    vector_backend: str,
    collection: str,
    batch_size: int,
    previous_manifest: dict[str, Any] | None = None,
    delete_stale: bool = True,
) -> dict[str, Any]:
    point_ids: list[str] = []
    indexed_count = 0
    dimensions = 0
    for start in range(0, len(documents), max(1, batch_size)):
        batch = documents[start : start + batch_size]
        embeddings = provider.embed([document.text for document in batch])
        points = []
        for document, embedding in zip(batch, embeddings, strict=True):
            point_id = point_id_for_chunk(collection, document.chunk_id)
            point_ids.append(point_id)
            dimensions = embedding.dimensions
            points.append(
                VectorPoint(
                    id=point_id,
                    vector=embedding.vector,
                    payload=vector_payload(document, mode, provider.provider, provider.model),
                )
            )
        if points:
            vector_store.ensure_collection(collection, dimensions)
            vector_store.upsert(collection, points)
            indexed_count += len(points)
    stale_point_ids = sorted(set((previous_manifest or {}).get("point_ids") or []) - set(point_ids))
    if delete_stale and stale_point_ids:
        vector_store.delete_points(collection, stale_point_ids)
    manifest = {
        "schema_version": VECTOR_INDEX_SCHEMA_VERSION,
        "mode": mode,
        "vector_backend": vector_backend,
        "collection": collection,
        "embedding_provider": provider.provider,
        "embedding_model": provider.model,
        "dimensions": dimensions,
        "document_count": len(documents),
        "point_count": len(point_ids),
        "point_ids": point_ids,
        "corpus_snapshot": corpus_snapshot(documents, provider.provider, provider.model),
    }
    return {
        "report_id": "vector_index_build_v1",
        "status": "indexed" if indexed_count == len(documents) else "partial",
        "mode": mode,
        "vector_backend": vector_backend,
        "collection": collection,
        "embedding_provider": provider.provider,
        "embedding_model": provider.model,
        "document_count": len(documents),
        "indexed_count": indexed_count,
        "dimensions": dimensions,
        "deleted_stale_count": len(stale_point_ids) if delete_stale else 0,
        "corpus_snapshot": manifest["corpus_snapshot"],
        "manifest": manifest,
        "boundary": (
            "The vector database is an ANN index over local CANON chunks, not the source of truth. "
            "Re-run indexing after corpus changes; deterministic point IDs update changed chunks."
        ),
    }


def vector_payload(
    document: RetrievalDocument,
    mode: str,
    provider: str,
    model: str,
) -> dict[str, Any]:
    return {
        "schema_version": VECTOR_INDEX_SCHEMA_VERSION,
        "mode": mode,
        "chunk_id": document.chunk_id,
        "work_id": document.work_id,
        "title": document.title,
        "source_name": document.source_name,
        "year": document.year,
        "section": document.section,
        "text_preview": document.text[:1200],
        "content_hash": text_hash(document.text),
        "embedding_provider": provider,
        "embedding_model": model,
    }


def corpus_snapshot(documents: list[RetrievalDocument], provider: str, model: str | None) -> str:
    digest = hashlib.sha256()
    digest.update(f"{provider}:{model or ''}\n".encode("utf-8"))
    for document in sorted(documents, key=lambda row: row.chunk_id):
        digest.update(document.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text_hash(document.text).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_path(data_dir: Path, mode: str, backend: str, provider: str, model: str | None) -> Path:
    return data_dir / "processed" / f"vector_index_{mode}_{safe_slug(backend)}_{artifact_key(provider, model)}.json"


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def public_index_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "manifest"} | {
        "manifest_point_count": len((report.get("manifest") or {}).get("point_ids") or [])
    }


def safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value.lower()).strip("-") or "value"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or refresh a CANON vector index.")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--embedding-provider", default="openrouter")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--vector-backend", default="qdrant")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--keep-stale", action="store_true")
    args = parser.parse_args()
    report = build_vector_index(
        mode=args.mode,
        embedding_provider_name=args.embedding_provider,
        embedding_model=args.embedding_model,
        vector_backend=args.vector_backend,
        collection=args.collection,
        batch_size=args.batch_size,
        delete_stale=not args.keep_stale,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
