from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.embeddings.store import load_embedding_records


def build_pgvector_plan(mode: str, provider: str = "local") -> dict:
    settings = load_settings()
    embeddings_path = settings.data_dir / "processed" / f"embeddings_{mode}_{provider}.jsonl"
    records = load_embedding_records(embeddings_path)
    dimensions = records[0]["dimensions"] if records else 0
    sql_path = settings.data_dir / "processed" / f"pgvector_{mode}_{provider}.sql"
    sql = render_pgvector_sql(records, dimensions)
    sql_path.write_text(sql, encoding="utf-8")
    report = {
        "mode": mode,
        "provider": provider,
        "embedding_path": str(embeddings_path),
        "sql_path": str(sql_path),
        "record_count": len(records),
        "dimensions": dimensions,
        "ready": bool(records),
    }
    write_json(settings.reports_dir / f"pgvector_{mode}_{provider}.json", report)
    return report


def render_pgvector_sql(records: list[dict], dimensions: int) -> str:
    lines = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        "CREATE TABLE IF NOT EXISTS chunk_embeddings (",
        "  chunk_id text PRIMARY KEY,",
        "  work_id text NOT NULL,",
        "  provider text NOT NULL,",
        "  model text NOT NULL,",
        f"  embedding vector({dimensions}) NOT NULL",
        ");",
        "TRUNCATE TABLE chunk_embeddings;",
    ]
    for record in records:
        vector = "[" + ",".join(str(round(float(value), 8)) for value in record["vector"]) + "]"
        lines.append(
            "INSERT INTO chunk_embeddings (chunk_id, work_id, provider, model, embedding) "
            f"VALUES ({quote(record['chunk_id'])}, {quote(record['work_id'])}, "
            f"{quote(record['provider'])}, {quote(record['model'])}, {quote(vector)});"
        )
    if dimensions:
        lines.append(
            "CREATE INDEX IF NOT EXISTS chunk_embeddings_embedding_hnsw "
            "ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);"
        )
    return "\n".join(lines) + "\n"


def quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a pgvector load plan for CANON embeddings.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--provider", default="local")
    args = parser.parse_args()
    print(json.dumps(build_pgvector_plan(args.mode, args.provider), indent=2))


if __name__ == "__main__":
    main()
