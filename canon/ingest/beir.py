from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.ingest.unstructured import ingest_unstructured_jsonl
from canon.product import report_io


DATASET_DESCRIPTIONS = {
    "scifact": "BEIR SciFact scientific claim verification benchmark.",
    "climate-fever": "BEIR Climate-FEVER real-world climate claim verification benchmark.",
    "nfcorpus": "BEIR NFCorpus biomedical and nutrition retrieval benchmark.",
}


DATASET_PROFILES = {
    "scifact": {
        "document_type": "academic_article",
        "domain": "scientific_claim_verification",
    },
    "nfcorpus": {
        "document_type": "biomedical_web_article",
        "domain": "biomedical_nutrition_retrieval",
    },
}


def import_beir_dataset(
    dataset_dir: Path,
    mode: str,
    benchmark_id: str | None = None,
    split: str = "test",
    max_documents: int | None = None,
    max_queries: int | None = None,
    include_qrels_documents: bool = False,
    include_title_in_chunks: bool = True,
    document_type: str | None = None,
    domain: str | None = None,
    chunk_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    dataset_dir = dataset_dir.resolve()
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    require_file(corpus_path)
    require_file(queries_path)
    require_file(qrels_path)

    selected_query_ids = selected_qrel_query_ids(qrels_path, max_queries=max_queries)
    required_doc_ids = (
        relevant_doc_ids(qrels_path, query_ids=selected_query_ids)
        if include_qrels_documents
        else set()
    )
    records = beir_corpus_to_unstructured(
        corpus_path,
        max_documents=max_documents,
        include_doc_ids=required_doc_ids,
        dataset_name=dataset_dir.name,
        include_title_in_chunks=include_title_in_chunks,
        document_type=document_type,
        domain=domain,
    )
    normalized_input = settings.data_dir / "raw" / f"beir_{mode}.jsonl"
    write_jsonl(normalized_input, records)
    ingest_report = ingest_unstructured_jsonl(
        normalized_input,
        mode=mode,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
    qrels = canonical_qrels(
        dataset_dir=dataset_dir,
        mode=mode,
        benchmark_id=benchmark_id or f"beir_{dataset_dir.name}_{split}",
        split=split,
        max_queries=max_queries,
    )
    qrels_output = settings.root / "gold" / f"{qrels['benchmark_id']}.json"
    report_io.write_json(qrels_output, qrels)
    report = {
        "report_id": "beir_import_v1",
        "dataset_dir": str(dataset_dir).replace("\\", "/"),
        "mode": mode,
        "split": split,
        "benchmark_id": qrels["benchmark_id"],
        "work_count": ingest_report["work_count"],
        "chunk_count": ingest_report["chunk_count"],
        "query_count": len(qrels["queries"]),
        "judgment_count": sum(len(query["relevant"]) for query in qrels["queries"]),
        "required_qrels_document_count": len(required_doc_ids),
        "include_title_in_chunks": include_title_in_chunks,
        "document_type": document_type,
        "domain": domain,
        "normalized_input": str(normalized_input).replace("\\", "/"),
        "qrels_path": str(qrels_output).replace("\\", "/"),
        "limitations": [
            "BEIR qrels are document-level; CANON expands each relevant document to all chunks from that document.",
            "Scores are comparable inside this CANON import, but not directly identical to official BEIR passage/document metrics.",
        ],
    }
    report_io.write_json(settings.reports_dir / f"beir_import_{mode}.json", report)
    return report


def beir_corpus_to_unstructured(
    path: Path,
    max_documents: int | None = None,
    include_doc_ids: set[str] | None = None,
    dataset_name: str | None = None,
    include_title_in_chunks: bool = True,
    document_type: str | None = None,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    include_doc_ids = include_doc_ids or set()
    profile = DATASET_PROFILES.get(str(dataset_name or path.parent.name).lower(), {})
    records = []
    seen_ids = set()
    for row in load_jsonl(path):
        doc_id = str(row.get("_id") or row.get("id") or "")
        if not doc_id or doc_id in seen_ids:
            continue
        if max_documents and len(records) >= max_documents and doc_id not in include_doc_ids:
            continue
        title = str(row.get("title") or doc_id)
        text = str(row.get("text") or row.get("content") or "")
        if not text.strip():
            continue
        chunk_text = titled_text(title, text) if include_title_in_chunks else text
        records.append(
            {
                "id": doc_id,
                "title": title,
                "text": chunk_text,
                "source_name": "BEIR",
                "document_type": document_type or profile.get("document_type", "academic_article"),
                "provenance": "public_benchmark",
                "domain": domain or profile.get("domain", "scientific_retrieval"),
                "metadata": {
                    "benchmark": "BEIR",
                    "beir_doc_id": doc_id,
                    "beir_dataset": dataset_name or path.parent.name,
                    "title_prefixed_for_retrieval": include_title_in_chunks,
                },
            }
        )
        seen_ids.add(doc_id)
    return records


def titled_text(title: str, text: str) -> str:
    clean_title = title.strip()
    clean_text = text.strip()
    if not clean_title or clean_title == clean_text:
        return clean_text
    if clean_text.lower().startswith(clean_title.lower()):
        return clean_text
    return f"{clean_title}\n\n{clean_text}"


def selected_qrel_query_ids(path: Path, max_queries: int | None = None) -> set[str]:
    query_ids = []
    for query_id in load_beir_qrels(path):
        query_ids.append(query_id)
        if max_queries and len(query_ids) >= max_queries:
            break
    return set(query_ids)


def relevant_doc_ids(path: Path, query_ids: set[str] | None = None) -> set[str]:
    qrels = load_beir_qrels(path)
    return {
        doc_id
        for query_id, docs in qrels.items()
        if query_ids is None or query_id in query_ids
        for doc_id, relevance in docs.items()
        if relevance > 0
    }


def canonical_qrels(
    dataset_dir: Path,
    mode: str,
    benchmark_id: str,
    split: str,
    max_queries: int | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    queries = load_beir_queries(dataset_dir / "queries.jsonl")
    doc_qrels = load_beir_qrels(dataset_dir / "qrels" / f"{split}.tsv")
    chunk_ids = chunks_by_work_id(settings.data_dir / "processed" / f"chunks_{mode}.json")
    rows = []
    for query_id, relevant_docs in sorted(doc_qrels.items()):
        expanded = {}
        for doc_id, relevance in relevant_docs.items():
            if relevance <= 0:
                continue
            for chunk_id in chunk_ids.get(doc_id, []):
                expanded[chunk_id] = relevance
        if not expanded:
            continue
        rows.append(
            {
                "id": query_id,
                "query": queries.get(query_id, query_id),
                "relevant": expanded,
                "metadata": {
                    "source_qrels": "beir_document_level",
                    "source_relevant_doc_count": sum(1 for value in relevant_docs.values() if value > 0),
                },
            }
        )
        if max_queries and len(rows) >= max_queries:
            break
    return {
        "benchmark_id": benchmark_id,
        "benchmark_kind": "public_beir_qrels_expanded_to_chunks",
        "description": DATASET_DESCRIPTIONS.get(dataset_dir.name, "BEIR public information retrieval benchmark."),
        "queries": rows,
    }


def load_beir_queries(path: Path) -> dict[str, str]:
    return {
        str(row.get("_id") or row.get("id")): str(row.get("text") or row.get("query") or "")
        for row in load_jsonl(path)
        if row.get("_id") or row.get("id")
    }


def load_beir_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = str(row.get("query-id") or row.get("query_id") or row.get("qid") or "")
            doc_id = str(row.get("corpus-id") or row.get("corpus_id") or row.get("docid") or "")
            if not query_id or not doc_id:
                continue
            qrels.setdefault(query_id, {})[doc_id] = float(row.get("score") or row.get("relevance") or 0)
    return qrels


def chunks_by_work_id(path: Path) -> dict[str, list[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, list[str]] = {}
    for row in rows:
        mapping.setdefault(str(row["work_id"]), []).append(str(row["id"]))
    return mapping


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected object JSONL row.")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required BEIR file not found: {path}")


def optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a BEIR-format dataset into CANON.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--include-qrels-documents", action="store_true")
    parser.add_argument("--no-title-prefix", action="store_true")
    parser.add_argument("--document-type", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--overlap-tokens", type=int, default=None)
    args = parser.parse_args()
    report = import_beir_dataset(
        dataset_dir=Path(args.dataset_dir),
        mode=args.mode,
        benchmark_id=args.benchmark_id,
        split=args.split,
        max_documents=optional_positive_int(args.max_documents, "max_documents"),
        max_queries=optional_positive_int(args.max_queries, "max_queries"),
        include_qrels_documents=args.include_qrels_documents,
        include_title_in_chunks=not args.no_title_prefix,
        document_type=args.document_type,
        domain=args.domain,
        chunk_tokens=optional_positive_int(args.chunk_tokens, "chunk_tokens"),
        overlap_tokens=args.overlap_tokens,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
