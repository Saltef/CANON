from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.eval.methods import evaluate_methods, load_method_set
from canon.eval.pipeline import compact_method_report, recommendations, technical_diagnostics


def parse_batch_sizes(value: str) -> list[int]:
    sizes = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        size = int(piece)
        if size <= 0:
            raise ValueError("Batch sizes must be positive integers.")
        sizes.append(size)
    if not sizes:
        raise ValueError("At least one batch size is required.")
    return sorted(set(sizes))


def run_incremental_batches(
    mode: str,
    batch_sizes: list[int],
    method_set_path: Path | None = None,
) -> dict:
    settings = load_settings()
    method_set_id = load_method_set(method_set_path)["method_set_id"]
    batch_reports = []
    for size in batch_sizes:
        method_report = evaluate_methods(mode, method_set_path, query_limit=size)
        compact = compact_method_report(method_report)
        diagnostics = technical_diagnostics(method_report, compact)
        batch_reports.append(
            {
                "batch_size": size,
                "actual_query_count": method_report["query_count"],
                "leaderboard": compact["leaderboard"],
                "technical_diagnostics": diagnostics,
                "recommendations": recommendations(compact, diagnostics),
            }
        )
    report = {
        "mode": mode,
        "method_set_id": method_set_id,
        "batch_sizes": batch_sizes,
        "batches": batch_reports,
        "stability": stability_report(batch_reports),
        "score_trajectories": score_trajectories(batch_reports),
        "recommendations": batch_recommendations(batch_reports),
    }
    write_json(settings.reports_dir / f"batch_eval_{mode}_{method_set_id}.json", report)
    return report


def stability_report(batch_reports: list[dict]) -> dict:
    top_methods = [
        batch["leaderboard"][0]["method_id"]
        for batch in batch_reports
        if batch["leaderboard"]
    ]
    top_changes = sum(
        1 for left, right in zip(top_methods, top_methods[1:], strict=False) if left != right
    )
    margins = [
        batch["technical_diagnostics"]["score_margin_top2"]
        for batch in batch_reports
        if batch["leaderboard"]
    ]
    return {
        "top_methods": top_methods,
        "top_change_count": top_changes,
        "stable_top_method": top_methods[-1] if top_methods and top_changes == 0 else None,
        "final_top_method": top_methods[-1] if top_methods else None,
        "minimum_top2_margin": min(margins) if margins else 0.0,
        "maximum_top2_margin": max(margins) if margins else 0.0,
    }


def score_trajectories(batch_reports: list[dict]) -> dict[str, list[dict]]:
    trajectories: dict[str, list[dict]] = {}
    for batch in batch_reports:
        size = batch["batch_size"]
        for row in batch["leaderboard"]:
            trajectories.setdefault(row["method_id"], []).append(
                {
                    "batch_size": size,
                    "rank": row["rank"],
                    "composite_score": row["composite_score"],
                    "claim_coverage": row["claim_coverage"],
                    "cluster_diversity": row["cluster_diversity"],
                    "context_relevance": row["context_relevance"],
                    "semantic_alignment": row["semantic_alignment"],
                    "conflict_awareness": row["conflict_awareness"],
                }
            )
    return trajectories


def batch_recommendations(batch_reports: list[dict]) -> list[str]:
    if not batch_reports:
        return ["No batches were evaluated."]
    stability = stability_report(batch_reports)
    recommendations = []
    if stability["top_change_count"] == 0:
        recommendations.append(
            f"Top method stayed stable across batches: {stability['final_top_method']}."
        )
    else:
        recommendations.append(
            "Top method changed across batches; expand query coverage before choosing a default."
        )
    if stability["minimum_top2_margin"] < 0.01:
        recommendations.append(
            "Top-two margin is narrow in at least one batch; run technical stress probes before declaring a winner."
        )
    final_batch = batch_reports[-1]
    if final_batch["technical_diagnostics"]["near_ties"]:
        recommendations.append(
            "Final batch still has near ties; compare tied methods on disagreement-specific and relevance-specific query slices."
        )
    return recommendations


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run incremental CANON method-eval batches.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--batch-sizes", default="1,3,5")
    parser.add_argument("--method-set", default=None)
    args = parser.parse_args()
    method_set_path = Path(args.method_set) if args.method_set else None
    print(
        json.dumps(
            run_incremental_batches(
                mode=args.mode,
                batch_sizes=parse_batch_sizes(args.batch_sizes),
                method_set_path=method_set_path,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
