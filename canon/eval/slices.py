from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.eval.harness import load_gold
from canon.eval.methods import (
    leaderboard,
    load_method_set,
    method_specs,
    method_summary,
    pairwise_summary,
    summarize_method_run,
    synthesize_for_method,
)
from canon.eval.pipeline import compact_method_report, recommendations, technical_diagnostics


TOPIC_KEYWORDS = {
    "democratic_peace": ["democracies", "democratic", "war"],
    "sanctions": ["sanctions"],
    "institutions": ["institutions"],
    "civil_war": ["civil war"],
    "human_rights": ["human rights"],
}


def parse_method_ids(value: str | None) -> list[str] | None:
    if not value:
        return None
    ids = [piece.strip() for piece in value.split(",") if piece.strip()]
    return ids or None


def run_stress_slices(
    mode: str,
    method_set_path: Path | None = None,
    method_ids: list[str] | None = None,
) -> dict:
    settings = load_settings()
    method_set = load_method_set(method_set_path)
    specs = method_specs(method_set)
    if method_ids:
        allowed = set(method_ids)
        specs = [spec for spec in specs if spec.id in allowed]
    queries = load_gold(settings.root / "gold" / "seed_queries.json")
    slices = build_slices(queries)
    slice_reports = []
    for slice_id, slice_queries in slices.items():
        if not slice_queries:
            continue
        method_report = evaluate_slice(mode, method_set["method_set_id"], slice_id, slice_queries, specs, method_set["score_weights"])
        compact = compact_method_report(method_report)
        diagnostics = technical_diagnostics(method_report, compact)
        slice_reports.append(
            {
                "slice_id": slice_id,
                "query_count": len(slice_queries),
                "query_ids": [query["id"] for query in slice_queries],
                "leaderboard": compact["leaderboard"],
                "technical_diagnostics": diagnostics,
                "recommendations": recommendations(compact, diagnostics),
            }
        )
    report = {
        "mode": mode,
        "method_set_id": method_set["method_set_id"],
        "method_ids": [spec.id for spec in specs],
        "slice_count": len(slice_reports),
        "slices": slice_reports,
        "cross_slice_summary": cross_slice_summary(slice_reports),
    }
    write_json(settings.reports_dir / f"stress_slices_{mode}_{method_set['method_set_id']}.json", report)
    return report


def build_slices(queries: list[dict]) -> dict[str, list[dict]]:
    slices: dict[str, list[dict]] = {"all": queries}
    for query in queries:
        slices.setdefault(f"type_{query['type']}", []).append(query)
    for topic, keywords in TOPIC_KEYWORDS.items():
        matched = []
        for query in queries:
            text = query["query"].lower()
            if any(keyword in text for keyword in keywords):
                matched.append(query)
        slices[f"topic_{topic}"] = matched
    return slices


def evaluate_slice(
    mode: str,
    method_set_id: str,
    slice_id: str,
    queries: list[dict],
    specs,
    score_weights: dict[str, float],
) -> dict:
    query_reports = []
    for query in queries:
        runs = [
            summarize_method_run(query, spec, synthesize_for_method(query["query"], spec, mode))
            for spec in specs
        ]
        query_reports.append(
            {
                "id": query["id"],
                "type": query["type"],
                "query": query["query"],
                "importance_expectation": query["importance_expectation"],
                "method_runs": runs,
            }
        )
    summary = method_summary(query_reports, specs, score_weights)
    report = {
        "mode": mode,
        "method_set_id": method_set_id,
        "slice_id": slice_id,
        "query_count": len(query_reports),
        "queries": query_reports,
        "method_summary": summary,
        "leaderboard": leaderboard(summary),
        "pairwise": pairwise_summary(query_reports, specs),
    }
    return report


def cross_slice_summary(slice_reports: list[dict]) -> dict:
    wins: dict[str, int] = {}
    near_ties: dict[str, int] = {}
    for report in slice_reports:
        if not report["leaderboard"]:
            continue
        top = report["leaderboard"][0]["method_id"]
        wins[top] = wins.get(top, 0) + 1
        for tie in report["technical_diagnostics"]["near_ties"]:
            near_ties[tie["left_method_id"]] = near_ties.get(tie["left_method_id"], 0) + 1
            near_ties[tie["right_method_id"]] = near_ties.get(tie["right_method_id"], 0) + 1
    return {
        "wins_by_method": dict(sorted(wins.items())),
        "near_tie_mentions_by_method": dict(sorted(near_ties.items())),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CANON technical stress-slice evaluation.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--method-set", default=None)
    parser.add_argument("--method-ids", default=None)
    args = parser.parse_args()
    method_set_path = Path(args.method_set) if args.method_set else None
    print(
        json.dumps(
            run_stress_slices(
                mode=args.mode,
                method_set_path=method_set_path,
                method_ids=parse_method_ids(args.method_ids),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
