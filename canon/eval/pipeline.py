from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.eval.methods import evaluate_methods


def run_evaluation_pipeline(
    mode: str,
    method_set_path: Path | None = None,
    query_limit: int | None = None,
) -> dict:
    settings = load_settings()
    method_report = evaluate_methods(mode, method_set_path, query_limit)
    compact = compact_method_report(method_report)
    diagnostics = technical_diagnostics(method_report, compact)
    report = {
        "mode": mode,
        "method_set_id": method_report["method_set_id"],
        "query_count": method_report["query_count"],
        "practice_dimensions": [
            "context_relevance",
            "citation_support",
            "evidence_coverage",
            "disagreement_awareness",
            "retrieval_overlap",
            "rank_stability",
            "metric_sensitivity",
        ],
        "leaderboard": compact["leaderboard"],
        "method_summary": compact["method_summary"],
        "pairwise_highlights": compact["pairwise_highlights"],
        "technical_diagnostics": diagnostics,
        "recommendations": recommendations(compact, diagnostics),
        "artifacts": {
            "full_method_eval": f"reports/method_eval_{mode}_{method_report['method_set_id']}.json",
            "evaluation_suite": f"reports/evaluation_suite_{mode}_{method_report['method_set_id']}.json",
        },
    }
    write_json(
        settings.reports_dir / f"evaluation_suite_{mode}_{method_report['method_set_id']}.json",
        report,
    )
    return report


def compact_method_report(method_report: dict) -> dict:
    return {
        "leaderboard": [
            {
                "rank": row["rank"],
                "method_id": row["method_id"],
                "label": row["label"],
                "composite_score": row["composite_score"],
                "claim_coverage": row["average_metrics"]["claim_coverage"],
                "cluster_diversity": row["average_metrics"]["cluster_diversity"],
                "context_relevance": row["average_metrics"]["context_relevance"],
                "semantic_alignment": row["average_metrics"]["semantic_alignment"],
                "conflict_awareness": row["average_metrics"]["conflict_awareness"],
            }
            for row in method_report["leaderboard"]
        ],
        "method_summary": {
            method_id: {
                "label": row["label"],
                "composite_score": row["composite_score"],
                "average_metrics": row["average_metrics"],
            }
            for method_id, row in method_report["method_summary"].items()
        },
        "pairwise_highlights": pairwise_highlights(method_report["pairwise"]),
    }


def pairwise_highlights(pairwise: list[dict]) -> dict:
    if not pairwise:
        return {
            "lowest_overlap": [],
            "largest_cluster_diversity_gain": [],
        }
    return {
        "lowest_overlap": sorted(
            pairwise,
            key=lambda row: row["average_retrieval_overlap"],
        )[:5],
        "largest_cluster_diversity_gain": sorted(
            pairwise,
            key=lambda row: row["average_cluster_diversity_delta_right_minus_left"],
            reverse=True,
        )[:5],
    }


def technical_diagnostics(method_report: dict, compact: dict) -> dict:
    leaderboard = compact["leaderboard"]
    return {
        "score_margin_top2": score_margin(leaderboard, 0, 1),
        "score_margin_top_to_last": score_margin(leaderboard, 0, -1),
        "near_ties": near_ties(leaderboard, tolerance=0.01),
        "metric_spread": metric_spread(method_report["method_summary"]),
        "retrieval_stability": retrieval_stability(method_report["pairwise"]),
        "dominance": dominance(method_report["method_summary"]),
    }


def score_margin(leaderboard: list[dict], left_index: int, right_index: int) -> float:
    if len(leaderboard) < 2:
        return 0.0
    try:
        return round(
            leaderboard[left_index]["composite_score"] - leaderboard[right_index]["composite_score"],
            6,
        )
    except IndexError:
        return 0.0


def near_ties(leaderboard: list[dict], tolerance: float) -> list[dict]:
    ties = []
    for left, right in zip(leaderboard, leaderboard[1:], strict=False):
        margin = round(left["composite_score"] - right["composite_score"], 6)
        if margin <= tolerance:
            ties.append(
                {
                    "left_method_id": left["method_id"],
                    "right_method_id": right["method_id"],
                    "margin": margin,
                }
            )
    return ties


def metric_spread(summary: dict[str, dict]) -> dict[str, dict]:
    metric_names = sorted(
        {
            metric
            for row in summary.values()
            for metric in row["average_metrics"]
            if isinstance(row["average_metrics"][metric], int | float)
        }
    )
    spread = {}
    for metric in metric_names:
        values = {
            method_id: row["average_metrics"].get(metric, 0.0)
            for method_id, row in summary.items()
        }
        high_method = max(values, key=values.get)
        low_method = min(values, key=values.get)
        spread[metric] = {
            "min": round(values[low_method], 6),
            "max": round(values[high_method], 6),
            "range": round(values[high_method] - values[low_method], 6),
            "max_method_id": high_method,
            "min_method_id": low_method,
        }
    return spread


def retrieval_stability(pairwise: list[dict]) -> dict:
    overlaps = [row["average_retrieval_overlap"] for row in pairwise]
    if not overlaps:
        return {"average_pairwise_overlap": 0.0, "min_pairwise_overlap": 0.0}
    return {
        "average_pairwise_overlap": round(sum(overlaps) / len(overlaps), 6),
        "min_pairwise_overlap": round(min(overlaps), 6),
    }


def dominance(summary: dict[str, dict]) -> list[dict]:
    rows = []
    items = list(summary.items())
    for left_id, left in items:
        for right_id, right in items:
            if left_id == right_id:
                continue
            left_metrics = left["average_metrics"]
            right_metrics = right["average_metrics"]
            comparable = [
                metric
                for metric in [
                    "claim_coverage",
                    "cluster_diversity",
                    "source_diversity",
                    "context_relevance",
                    "semantic_alignment",
                    "citation_support",
                    "conflict_awareness",
                ]
                if metric in left_metrics and metric in right_metrics
            ]
            if comparable and all(left_metrics[m] >= right_metrics[m] for m in comparable):
                if any(left_metrics[m] > right_metrics[m] for m in comparable):
                    rows.append({"dominant_method_id": left_id, "dominated_method_id": right_id})
    return rows


def recommendations(compact: dict, diagnostics: dict) -> list[str]:
    leaderboard = compact["leaderboard"]
    if not leaderboard:
        return ["No method runs were available."]
    top = leaderboard[0]
    recommendations = [
        f"Use {top['method_id']} as the current default candidate; it has the top composite score.",
    ]
    if top["conflict_awareness"] < 0.5:
        recommendations.append(
            "Improve disagreement surfacing; even the leading method rarely attaches conflict notes."
        )
    if top["context_relevance"] < 0.7:
        recommendations.append(
            "Add reranking or query decomposition; the leading method is gaining diversity at some relevance cost."
        )
    if diagnostics["near_ties"]:
        recommendations.append(
            "Treat near-tied methods as unresolved until more technical probes separate them."
        )
    return recommendations


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CANON method evaluation pipeline.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--method-set", default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    args = parser.parse_args()
    method_set_path = Path(args.method_set) if args.method_set else None
    print(
        json.dumps(
            run_evaluation_pipeline(args.mode, method_set_path, args.query_limit),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
