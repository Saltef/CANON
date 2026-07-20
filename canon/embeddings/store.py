from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.embeddings.providers import EmbeddingResult, get_embedding_provider
from canon.retrieval.clusters import load_cluster_assignments
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus


def build_embedding_store(
    mode: str,
    provider_name: str = "local",
    model: str | None = None,
    batch_size: int = 32,
) -> dict:
    settings = load_settings()
    documents = load_processed_corpus(
        settings.data_dir,
        mode,
        cluster_assignments=load_cluster_assignments(settings.data_dir, mode),
    )
    provider = get_embedding_provider(provider_name, model)
    records = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        embeddings = provider.embed([document.text for document in batch])
        records.extend(
            record_for_embedding(document, embedding)
            for document, embedding in zip(batch, embeddings, strict=True)
        )
    output_path = embedding_store_path(settings.data_dir, mode, provider.provider, provider.model)
    write_jsonl(output_path, records)
    report = {
        "mode": mode,
        "provider": provider.provider,
        "model": provider.model,
        "document_count": len(documents),
        "embedding_count": len(records),
        "dimensions": records[0]["dimensions"] if records else 0,
        "output_path": str(output_path),
    }
    write_json(settings.reports_dir / f"embeddings_{mode}_{artifact_key(provider.provider, provider.model)}.json", report)
    return report


def embedding_store_path(data_dir: Path, mode: str, provider: str, model: str | None = None) -> Path:
    return data_dir / "processed" / f"embeddings_{mode}_{artifact_key(provider, model)}.jsonl"


def artifact_key(provider: str, model: str | None = None) -> str:
    if not model:
        return safe_slug(provider)
    return f"{safe_slug(provider)}_{safe_slug(model)}"


def safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value.lower()).strip("-") or "model"


def record_for_embedding(document: RetrievalDocument, embedding: EmbeddingResult) -> dict:
    return {
        "chunk_id": document.chunk_id,
        "work_id": document.work_id,
        "provider": embedding.provider,
        "model": embedding.model,
        "dimensions": embedding.dimensions,
        "vector": embedding.vector,
    }


def load_embedding_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CANON embedding store.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(build_embedding_store(args.mode, args.provider, args.model, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
