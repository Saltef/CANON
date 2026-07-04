from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from canon.config import load_settings


def build_data_card(mode: str) -> dict:
    settings = load_settings()
    works = load_json(settings.data_dir / "processed" / f"works_{mode}.json", [])
    chunks = load_json(settings.data_dir / "processed" / f"chunks_{mode}.json", [])
    corpus = load_json(settings.reports_dir / f"corpus_{mode}.json", {})
    quality = load_json(settings.reports_dir / f"quality_diagnostics_{mode}.json", {})
    card = {
        "mode": mode,
        "corpus_id": corpus.get("corpus_id", mode),
        "source_modes": corpus.get("source_modes", []),
        "work_count": len(works),
        "chunk_count": len(chunks),
        "topic_mode_counts": topic_mode_counts(works),
        "coverage": coverage(works, chunks),
        "year_distribution": dict(sorted(Counter(str(work.get("year")) for work in works if work.get("year")).items())),
        "top_sources": dict(Counter(work.get("source_name") or "UNKNOWN" for work in works).most_common(20)),
        "openalex_filters": inferred_openalex_filters(settings),
        "quality_signal_summary": quality.get("signals", {}),
        "limitations": limitations(works, chunks, quality),
    }
    write_json(settings.reports_dir / f"data_card_{mode}.json", card)
    write_markdown(settings.reports_dir / f"data_card_{mode}.md", card)
    return card


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def topic_mode_counts(works: list[dict]) -> dict:
    counts = Counter()
    for work in works:
        modes = work.get("corpus_source_modes") or []
        for mode in modes:
            counts[mode] += 1
    return dict(sorted(counts.items()))


def coverage(works: list[dict], chunks: list[dict]) -> dict:
    total = len(works)
    return {
        "abstract_coverage": ratio(sum(1 for work in works if work.get("abstract")), total),
        "open_access_coverage": ratio(sum(1 for work in works if work.get("is_open_access")), total),
        "pdf_url_coverage": ratio(sum(1 for work in works if work.get("pdf_url")), total),
        "reference_list_coverage": ratio(sum(1 for work in works if int(work.get("referenced_work_count") or 0) > 0), total),
        "authorship_coverage": ratio(sum(1 for work in works if work.get("author_display_names")), total),
        "author_impact_coverage": ratio(sum(1 for work in works if int(work.get("max_author_cited_by_count") or 0) > 0), total),
        "chunks_per_work": round(len(chunks) / total, 6) if total else 0.0,
    }


def inferred_openalex_filters(settings) -> dict:
    raw = settings.raw.get("openalex", {})
    return {
        "from_publication_year": raw.get("from_publication_year"),
        "to_publication_year": raw.get("to_publication_year"),
        "default_per_page": raw.get("per_page"),
    }


def limitations(works: list[dict], chunks: list[dict], quality: dict) -> list[str]:
    limits = []
    cov = coverage(works, chunks)
    if cov["pdf_url_coverage"] < 0.5:
        limits.append("PDF URL coverage is low; full-text coverage is not representative yet.")
    if cov["reference_list_coverage"] < 0.8:
        limits.append("Reference-list coverage is incomplete; citation graph signals may understate some works.")
    if cov["author_impact_coverage"] < 0.8:
        limits.append("Author-impact coverage is incomplete; author-score comparisons should be treated as partial.")
    if len(works) < 100:
        limits.append("Corpus is small; results are suitable for technical testing, not field-level claims.")
    if quality and int(quality.get("chunk_count") or 0) != len(chunks):
        limits.append("Quality diagnostics chunk count differs from current processed chunks.")
    return limits


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, card: dict) -> None:
    lines = [
        f"# CANON Data Card: {card['mode']}",
        "",
        f"- Works: {card['work_count']}",
        f"- Chunks: {card['chunk_count']}",
        f"- Source modes: {', '.join(card['source_modes']) if card['source_modes'] else 'unknown'}",
        "",
        "## Coverage",
        "",
    ]
    for key, value in card["coverage"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Limitations", ""])
    for item in card["limitations"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a corpus data card for CANON.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    args = parser.parse_args()
    print(json.dumps(build_data_card(args.mode), indent=2))


if __name__ == "__main__":
    main()
