from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path

from canon.config import load_settings
from canon.db.sql_export import export_load_sql
from canon.ingest.chunker import chunk_work_abstract
from canon.ingest.diagnostics import corpus_diagnostics, serialize_works
from canon.ingest.openalex import (
    enrich_works_with_author_metrics,
    harvest_openalex,
    load_fixture,
    normalize_work,
)
from canon.ingest.pdf_resolver import pdf_candidates


def load_topic_file(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def topic_scope(topic: dict) -> tuple[list[str], list[str]]:
    if "scope" in topic:
        return (
            list(topic["scope"].get("openalex_concept_ids", [])),
            list(topic["scope"].get("seed_terms", [])),
        )
    openalex = topic.get("openalex", {})
    search = openalex.get("search") or topic.get("topic", {}).get("query")
    return (list(openalex.get("concept_ids", [])), [search] if search else [])


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(
    dry_run: bool,
    live: bool,
    max_results: int | None,
    enrich_authors: bool = False,
    topic_file: str | None = None,
    output_mode: str | None = None,
) -> dict:
    settings = load_settings()
    raw = settings.raw
    topic_path = settings.root / (topic_file or raw["openalex"]["topic_file"])
    topic = load_topic_file(topic_path)
    topic_ids, search_terms = topic_scope(topic)
    target = max_results or int(raw["openalex"]["max_results"])

    if live and dry_run:
        raise ValueError("Choose either --dry-run or --live, not both.")
    if live:
        raw_works = harvest_openalex(raw, topic_ids, search_terms, target)
        if enrich_authors:
            raw_works = enrich_works_with_author_metrics(
                raw_works,
                base_url=raw["openalex"]["base_url"],
                mailto=raw["project"]["contact_email"],
                api_key=os.getenv(raw["openalex"].get("api_key_env") or ""),
            )
    else:
        raw_works = load_fixture(settings.fixtures_dir / "openalex_sample.json")

    works = [normalize_work(item) for item in raw_works]
    chunk_tokens = int(raw["ingest"]["chunk_tokens"])
    overlap_tokens = int(raw["ingest"]["chunk_overlap_tokens"])
    chunks = [
        chunk
        for work in works
        for chunk in chunk_work_abstract(work, chunk_tokens, overlap_tokens)
    ]
    pdfs = [candidate.__dict__ for work in works for candidate in pdf_candidates(work)]

    mode = output_mode or ("live" if live else "dry_run")
    write_json(settings.data_dir / "raw" / f"openalex_{mode}.json", raw_works)
    write_json(settings.data_dir / "processed" / f"works_{mode}.json", serialize_works(works))
    write_json(settings.data_dir / "processed" / f"chunks_{mode}.json", [chunk.__dict__ for chunk in chunks])
    write_json(settings.data_dir / "processed" / f"pdf_candidates_{mode}.json", pdfs)
    diagnostics = corpus_diagnostics(works)
    report = {
        "mode": mode,
        "topic_file": str(topic_path),
        "author_enriched": bool(live and enrich_authors),
        "work_count": len(works),
        "chunk_count": len(chunks),
        "pdf_candidate_count": len(pdfs),
        "diagnostics": diagnostics,
    }
    write_json(settings.reports_dir / f"phase1_ingest_{mode}.json", report)
    sql_path = settings.data_dir / "processed" / f"load_{mode}.sql"
    sql_path.write_text(export_load_sql(works, chunks, mode, diagnostics), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CANON Phase 1 ingestion.")
    parser.add_argument("--dry-run", action="store_true", help="Use local fixture data.")
    parser.add_argument("--live", action="store_true", help="Harvest live OpenAlex data.")
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--enrich-authors", action="store_true")
    parser.add_argument("--topic-file", default=None)
    parser.add_argument("--output-mode", default=None)
    args = parser.parse_args()
    report = run(
        dry_run=args.dry_run or not args.live,
        live=args.live,
        max_results=args.max_results,
        enrich_authors=args.enrich_authors,
        topic_file=args.topic_file,
        output_mode=args.output_mode,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
