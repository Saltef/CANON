from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.corpus.packs import build_topic_pack_manifest


def build_expansion_report(
    mode: str,
    pack_id: str = "social_science_ir",
    profile_path: str | None = None,
    target_work_count: int | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    profile_file = settings.root / (profile_path or "conf/corpus_sources.json")
    profile = json.loads(profile_file.read_text(encoding="utf-8"))
    pack = build_topic_pack_manifest(max_results=per_topic_target(profile, target_work_count))
    works = load_json(settings.data_dir / "processed" / f"works_{mode}.json", [])
    chunks = load_json(settings.data_dir / "processed" / f"chunks_{mode}.json", [])
    raw = load_json(settings.data_dir / "raw" / f"openalex_{mode}.json", [])
    target = int(target_work_count or profile.get("target_work_count") or 1000)

    coverage = corpus_coverage(works, chunks)
    source_modes = sorted(
        {
            source_mode
            for work in works
            for source_mode in (work.get("corpus_source_modes") or [])
        }
    )
    report = {
        "mode": mode,
        "profile_id": profile["profile_id"],
        "primary_index": profile["primary_index"],
        "target_work_count": target,
        "current_work_count": len(works),
        "current_chunk_count": len(chunks),
        "source_modes": source_modes,
        "topic_count": pack["topic_count"],
        "source_registry": profile["sources"],
        "implemented_sources": [
            source for source in profile["sources"] if source.get("status") == "implemented"
        ],
        "planned_sources": [
            source for source in profile["sources"] if source.get("status") != "implemented"
        ],
        "coverage": coverage,
        "topic_balance": topic_balance(works),
        "index_database_usage": index_database_usage(raw, works),
        "recommended_harvest_commands": recommended_harvest_commands(pack, target),
        "readiness": expansion_readiness(profile, target, works, coverage),
    }
    write_json(settings.reports_dir / f"corpus_expansion_{mode}.json", report)
    return report


def per_topic_target(profile: dict[str, Any], target_work_count: int | None) -> int:
    target = int(target_work_count or profile.get("target_work_count") or 1000)
    topic_count = int(profile.get("minimum_topic_count") or 1)
    return max(1, target // topic_count)


def corpus_coverage(works: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, float]:
    total = len(works)
    return {
        "abstract_coverage": ratio(sum(1 for work in works if work.get("abstract")), total),
        "open_access_coverage": ratio(sum(1 for work in works if work.get("is_open_access")), total),
        "pdf_url_coverage": ratio(sum(1 for work in works if work.get("pdf_url")), total),
        "reference_coverage": ratio(
            sum(1 for work in works if int(work.get("referenced_work_count") or 0) > 0),
            total,
        ),
        "authorship_coverage": ratio(sum(1 for work in works if work.get("author_display_names")), total),
        "author_index_coverage": ratio(
            sum(1 for work in works if int(work.get("max_author_cited_by_count") or 0) > 0),
            total,
        ),
        "chunks_per_work": round(len(chunks) / total, 6) if total else 0.0,
    }


def topic_balance(works: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for work in works:
        for mode in work.get("corpus_source_modes") or ["unknown"]:
            counts[mode] += 1
    values = list(counts.values())
    return {
        "counts": dict(sorted(counts.items())),
        "min_topic_count": min(values, default=0),
        "max_topic_count": max(values, default=0),
    }


def index_database_usage(raw_works: list[dict[str, Any]], works: list[dict[str, Any]]) -> dict[str, Any]:
    openalex_author_records = 0
    author_ids = set()
    for raw in raw_works:
        for authorship in raw.get("authorships") or []:
            author = authorship.get("author") or {}
            if author.get("id"):
                author_ids.add(author["id"])
            if int(author.get("cited_by_count") or 0) > 0 or int(author.get("works_count") or 0) > 0:
                openalex_author_records += 1
    return {
        "primary_index": "OpenAlex",
        "author_index": "OpenAlex Authors API",
        "unique_author_ids": len(author_ids),
        "author_records_with_index_metrics": openalex_author_records,
        "works_with_author_score_inputs": sum(
            1 for work in works if int(work.get("max_author_cited_by_count") or 0) > 0
        ),
        "work_index_fields_used": [
            "id",
            "doi",
            "publication_year",
            "abstract_inverted_index",
            "cited_by_count",
            "referenced_works",
            "authorships",
            "open_access",
            "primary_location",
            "best_oa_location",
        ],
    }


def recommended_harvest_commands(pack: dict[str, Any], target_work_count: int) -> list[str]:
    per_topic = max(1, int(target_work_count * 1.1) // max(1, pack["topic_count"]))
    return [
        (
            "python -m canon.corpus.build "
            "--corpus-id social_science_ir_10k "
            f"--harvest --max-results {per_topic} --corpus-only "
            "--top-k 5 --policies lexical,balanced,semantic,rag,diverse,conflict_aware"
        ),
        (
            "python -m canon.corpus.expansion "
            "--mode social_science_ir_10k "
            f"--target-work-count {target_work_count}"
        ),
    ]


def expansion_readiness(
    profile: dict[str, Any],
    target_work_count: int,
    works: list[dict[str, Any]],
    coverage: dict[str, float],
) -> dict[str, Any]:
    checks = [
        check("work_count_target", len(works) >= target_work_count, f"{len(works)} / {target_work_count} works"),
        check(
            "abstract_coverage",
            coverage["abstract_coverage"] >= float(profile["minimum_abstract_coverage"]),
            str(coverage["abstract_coverage"]),
        ),
        check(
            "reference_coverage",
            coverage["reference_coverage"] >= float(profile["minimum_reference_coverage"]),
            str(coverage["reference_coverage"]),
        ),
        check(
            "author_index_coverage",
            coverage["author_index_coverage"] >= float(profile["minimum_author_index_coverage"]),
            str(coverage["author_index_coverage"]),
        ),
    ]
    return {
        "status": "pass" if all(item["passed"] for item in checks) else "needs_expansion",
        "checks": checks,
    }


def check(check_id: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "evidence": evidence}


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CANON corpus expansion report.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--target-work-count", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            build_expansion_report(
                mode=args.mode,
                profile_path=args.profile,
                target_work_count=args.target_work_count,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
