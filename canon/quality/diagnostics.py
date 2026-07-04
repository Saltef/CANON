from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

from canon.config import load_settings
from canon.retrieval.corpus import load_processed_corpus


SIGNALS = [
    "source_quality",
    "citation_centrality",
    "author_score",
    "reference_coverage",
    "section_role",
    "claim_density",
    "recency",
]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(mean(values), 6),
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return round(numerator / (denom_x * denom_y), 6)


def signal_value(document, signal: str) -> float:
    if signal in document.work_signals:
        return float(document.work_signals[signal])
    return float(document.chunk_importance.get(signal, 0.0))


def quality_diagnostics(mode: str) -> dict:
    settings = load_settings()
    documents = load_processed_corpus(settings.data_dir, mode)
    values = {
        signal: [signal_value(document, signal) for document in documents]
        for signal in SIGNALS
    }
    correlations = {}
    for left in SIGNALS:
        for right in SIGNALS:
            if left >= right:
                continue
            correlations[f"{left}__{right}"] = pearson(values[left], values[right])
    report = {
        "mode": mode,
        "chunk_count": len(documents),
        "work_count": len({document.work_id for document in documents}),
        "signals": {signal: summarize(signal_values) for signal, signal_values in values.items()},
        "correlations": correlations,
        "top_sources": dict(Counter(document.source_name or "UNKNOWN" for document in documents).most_common(20)),
        "year_distribution": dict(sorted(Counter(str(document.year) for document in documents).items())),
    }
    write_json(settings.reports_dir / f"quality_diagnostics_{mode}.json", report)
    return report


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality and importance diagnostics.")
    parser.add_argument("--mode", default="dry_run")
    args = parser.parse_args()
    print(json.dumps(quality_diagnostics(args.mode), indent=2))


if __name__ == "__main__":
    main()
