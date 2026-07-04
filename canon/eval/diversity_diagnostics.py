from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from canon.config import load_settings
from canon.eval.diversity_gate import load_diversity_audit


DEFAULT_METHOD_SET_ID = "baseline_methods_v1"
DEFAULT_DIVERSE_METHOD_ID = "diverse_k5_template"
DEFAULT_BASELINE_METHOD_ID = "lexical_k5_template"


def run_diversity_diagnostics(
    mode: str,
    method_set_id: str = DEFAULT_METHOD_SET_ID,
    diverse_method_id: str = DEFAULT_DIVERSE_METHOD_ID,
    baseline_method_id: str = DEFAULT_BASELINE_METHOD_ID,
) -> dict:
    settings = load_settings()
    audit = load_diversity_audit(
        settings.reports_dir,
        mode,
        method_set_id,
        diverse_method_id,
        baseline_method_id,
    )
    query_diagnostics = [diagnose_query(query) for query in audit.get("queries", [])]
    report = {
        "mode": mode,
        "method_set_id": method_set_id,
        "diverse_method_id": diverse_method_id,
        "baseline_method_id": baseline_method_id,
        "query_count": len(query_diagnostics),
        "headline": headline(query_diagnostics),
        "aggregate": aggregate_diagnostics(query_diagnostics),
        "queries": query_diagnostics,
        "next_actions": next_actions(query_diagnostics, diverse_method_id),
    }
    write_json(
        settings.reports_dir
        / f"diversity_diagnostics_{mode}_{method_set_id}_{diverse_method_id}_vs_{baseline_method_id}.json",
        report,
    )
    return report


def diagnose_query(query: dict) -> dict:
    added = query.get("added_titles", [])
    removed = query.get("removed_titles", [])
    verdict = query.get("verdict", "")
    factors = diagnostic_factors(query, added, removed)
    failure_mode = classify_failure_mode(query, factors)
    return {
        "query_id": query["query_id"],
        "query_type": query.get("query_type"),
        "query": query.get("query"),
        "verdict": verdict,
        "failure_mode": failure_mode,
        "factor_summary": factors,
        "risk_signals": risk_signals(query, factors),
        "suspect_additions": suspect_additions(added),
        "strong_additions": strong_additions(added),
        "lost_high_focus_items": lost_high_focus_items(removed, added),
        "recommendation": recommendation_for(failure_mode, query, factors),
    }


def diagnostic_factors(query: dict, added: list[dict], removed: list[dict]) -> dict:
    added_focus = [float(item.get("focus_coverage") or 0.0) for item in added]
    added_relevance = [float(item.get("relevance") or 0.0) for item in added]
    removed_focus = [float(item.get("focus_coverage") or 0.0) for item in removed]
    removed_relevance = [float(item.get("relevance") or 0.0) for item in removed]
    return {
        "focus_term_count": len(query.get("focus_terms") or []),
        "cluster_delta": int(query.get("cluster_delta") or 0),
        "rank_overlap": float(query.get("rank_overlap") or 0.0),
        "added_count": int(query.get("added_count") or len(added)),
        "useful_breadth_count": int(query.get("useful_breadth_count") or 0),
        "noise_addition_count": int(query.get("noise_addition_count") or 0),
        "breadth_precision": float(query.get("breadth_precision") or 0.0),
        "noise_rate": float(query.get("noise_rate") or 0.0),
        "average_added_focus_coverage": average(added_focus),
        "average_removed_focus_coverage": average(removed_focus),
        "average_added_relevance": average(added_relevance),
        "average_removed_relevance": average(removed_relevance),
        "focus_coverage_delta": round(average(added_focus) - average(removed_focus), 6),
        "relevance_delta": round(average(added_relevance) - average(removed_relevance), 6),
    }


def classify_failure_mode(query: dict, factors: dict) -> str:
    if query.get("query_type") == "off_topic" and factors["cluster_delta"] > 0:
        return "off_topic_breadth"
    if factors["cluster_delta"] <= 0:
        return "no_new_cluster_gain"
    if factors["noise_rate"] >= 0.5:
        return "noisy_cluster_expansion"
    if factors["breadth_precision"] >= 0.5 and factors["noise_rate"] <= 0.25:
        return "successful_cluster_breadth"
    if factors["average_added_focus_coverage"] < 0.25:
        return "low_focus_coverage"
    if factors["relevance_delta"] < -0.2:
        return "relevance_breadth_tradeoff"
    if factors["focus_coverage_delta"] < -0.2:
        return "lost_more_focused_items"
    return "mixed_cluster_breadth"


def risk_signals(query: dict, factors: dict) -> list[str]:
    signals = []
    if query.get("query_type") == "off_topic":
        signals.append("off_topic_query")
    if factors["focus_term_count"] <= 3:
        signals.append("underspecified_focus_terms")
    if factors["rank_overlap"] < 0.3:
        signals.append("large_rank_shift")
    if factors["average_added_focus_coverage"] < 0.25 and factors["added_count"] > 0:
        signals.append("low_added_focus_coverage")
    if factors["relevance_delta"] < -0.2:
        signals.append("added_items_less_relevant_than_removed_items")
    if factors["noise_rate"] > 0.0:
        signals.append("contains_estimated_noise")
    return signals


def suspect_additions(added: list[dict], limit: int = 5) -> list[dict]:
    suspects = [
        item
        for item in added
        if float(item.get("focus_coverage") or 0.0) == 0.0
        or float(item.get("relevance") or 0.0) < 0.35
    ]
    return compact_items(sorted(suspects, key=addition_risk_key, reverse=True)[:limit])


def strong_additions(added: list[dict], limit: int = 5) -> list[dict]:
    strong = [
        item
        for item in added
        if float(item.get("focus_coverage") or 0.0) >= 0.25
        and float(item.get("relevance") or 0.0) >= 0.5
    ]
    return compact_items(
        sorted(
            strong,
            key=lambda item: (
                float(item.get("focus_coverage") or 0.0),
                float(item.get("relevance") or 0.0),
            ),
            reverse=True,
        )[:limit]
    )


def lost_high_focus_items(removed: list[dict], added: list[dict], limit: int = 5) -> list[dict]:
    if not removed:
        return []
    best_added_focus = max((float(item.get("focus_coverage") or 0.0) for item in added), default=0.0)
    lost = [
        item
        for item in removed
        if float(item.get("focus_coverage") or 0.0) > best_added_focus
        and float(item.get("relevance") or 0.0) >= 0.5
    ]
    return compact_items(
        sorted(
            lost,
            key=lambda item: (
                float(item.get("focus_coverage") or 0.0),
                float(item.get("relevance") or 0.0),
            ),
            reverse=True,
        )[:limit]
    )


def recommendation_for(failure_mode: str, query: dict, factors: dict) -> str:
    if failure_mode == "successful_cluster_breadth":
        return "Keep diversity pressure for this query class; added clusters preserve enough query focus."
    if failure_mode == "off_topic_breadth":
        return "Apply weak-support/off-topic gating before diversity expansion and avoid treating added breadth as support."
    if failure_mode == "no_new_cluster_gain":
        return "Do not increase diversity weight; baseline already covers comparable clusters for this query."
    if failure_mode == "noisy_cluster_expansion":
        return "Require a focus-coverage or relevance floor before accepting new-cluster additions."
    if failure_mode == "low_focus_coverage":
        return "Use query decomposition or stricter focus-term matching before adding diversity bonus."
    if failure_mode == "relevance_breadth_tradeoff":
        return "Tune diversity as a post-filter constrained by minimum relevance instead of a broad rank bonus."
    if failure_mode == "lost_more_focused_items":
        return "Preserve high-focus baseline items, then fill remaining slots with diverse clusters."
    if query.get("query_type") in {"comparison", "debate_mapping", "conflict"}:
        return "Use moderate diversity pressure, but inspect added clusters for paradigm/topic coverage."
    if factors["focus_term_count"] <= 3:
        return "Ask for a more specific query or expand focus terms with domain synonyms before diversity selection."
    return "Treat as mixed breadth and require explanation-level inspection before using in synthesis."


def aggregate_diagnostics(query_diagnostics: list[dict]) -> dict:
    by_verdict: dict[str, dict] = {}
    by_type: dict[str, dict] = {}
    by_failure_mode: dict[str, int] = dict(
        sorted(Counter(item["failure_mode"] for item in query_diagnostics).items())
    )
    for key, group in group_by(query_diagnostics, "verdict").items():
        by_verdict[key] = aggregate_group(group)
    for key, group in group_by(query_diagnostics, "query_type").items():
        by_type[key] = aggregate_group(group)
    return {
        "by_verdict": by_verdict,
        "by_query_type": by_type,
        "failure_mode_counts": by_failure_mode,
        "risk_signal_counts": dict(
            sorted(Counter(signal for item in query_diagnostics for signal in item["risk_signals"]).items())
        ),
    }


def aggregate_group(group: list[dict]) -> dict:
    factors = [item["factor_summary"] for item in group]
    return {
        "query_count": len(group),
        "average_cluster_delta": average(item["cluster_delta"] for item in factors),
        "average_breadth_precision": average(item["breadth_precision"] for item in factors),
        "average_noise_rate": average(item["noise_rate"] for item in factors),
        "average_added_focus_coverage": average(
            item["average_added_focus_coverage"] for item in factors
        ),
        "average_relevance_delta": average(item["relevance_delta"] for item in factors),
        "query_ids": [item["query_id"] for item in group],
    }


def headline(query_diagnostics: list[dict]) -> dict:
    counts = Counter(item["verdict"] for item in query_diagnostics)
    return {
        "finding": "Diversity-first is conditional: it helps most when new clusters retain query focus.",
        "useful_breadth_queries": counts.get("useful_breadth", 0),
        "mixed_or_risky_queries": (
            counts.get("mixed_breadth", 0)
            + counts.get("off_topic_breadth_risk", 0)
            + counts.get("noisy_breadth", 0)
        ),
        "no_gain_queries": counts.get("no_diversity_gain", 0),
        "claim_language": (
            "Report diversity-first retrieval as a breadth mechanism with focus-gated benefits, "
            "not as a uniformly superior method."
        ),
    }


def next_actions(query_diagnostics: list[dict], diverse_method_id: str = "") -> list[dict]:
    actions = []
    failure_counts = Counter(item["failure_mode"] for item in query_diagnostics)
    if failure_counts.get("low_focus_coverage", 0) or failure_counts.get("relevance_breadth_tradeoff", 0):
        if "focus_diverse" in diverse_method_id:
            actions.append(
                {
                    "id": "validate_focus_gates",
                    "priority": "high",
                    "action": "Keep focus-gated diversity in the candidate set and tune floors on a frozen validation split.",
                }
            )
        else:
            actions.append(
                {
                    "id": "focus_gated_diversity",
                    "priority": "high",
                    "action": "Add a diversity selector that only grants cluster bonus above a relevance/focus floor.",
                }
            )
    if failure_counts.get("off_topic_breadth", 0):
        actions.append(
            {
                "id": "off_topic_guardrail",
                "priority": "high",
                "action": "Run weak-support/off-topic detection before diversity expansion.",
            }
        )
    if failure_counts.get("lost_more_focused_items", 0):
        actions.append(
            {
                "id": "preserve_focused_baseline",
                "priority": "medium",
                "action": "Reserve top slots for high-focus baseline evidence before filling breadth slots.",
            }
        )
    actions.append(
        {
            "id": "query_type_policy",
            "priority": "medium",
            "action": "Learn query-type defaults: more diversity for comparison/debate, less for off-topic or narrow synthesis.",
        }
    )
    return actions


def group_by(items: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(str(item.get(key) or "UNKNOWN"), []).append(item)
    return dict(sorted(groups.items()))


def addition_risk_key(item: dict) -> tuple[float, float]:
    return (
        1.0 - float(item.get("focus_coverage") or 0.0),
        1.0 - float(item.get("relevance") or 0.0),
    )


def compact_items(items: list[dict]) -> list[dict]:
    return [
        {
            "rank": item.get("rank"),
            "chunk_id": item.get("chunk_id"),
            "title": item.get("title"),
            "cluster_id": item.get("cluster_id"),
            "relevance": item.get("relevance"),
            "semantic_similarity": item.get("semantic_similarity"),
            "focus_coverage": item.get("focus_coverage"),
        }
        for item in items
    ]


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cli_summary(report: dict) -> dict:
    return {
        "mode": report["mode"],
        "method_set_id": report["method_set_id"],
        "query_count": report["query_count"],
        "headline": report["headline"],
        "aggregate": report["aggregate"],
        "next_actions": report["next_actions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose when diversity-first retrieval helps or hurts.")
    parser.add_argument("--mode", default="social_science_ir_10k")
    parser.add_argument("--method-set-id", default=DEFAULT_METHOD_SET_ID)
    parser.add_argument("--diverse-method-id", default=DEFAULT_DIVERSE_METHOD_ID)
    parser.add_argument("--baseline-method-id", default=DEFAULT_BASELINE_METHOD_ID)
    args = parser.parse_args()
    print(
        json.dumps(
            cli_summary(
                run_diversity_diagnostics(
                    mode=args.mode,
                    method_set_id=args.method_set_id,
                    diverse_method_id=args.diverse_method_id,
                    baseline_method_id=args.baseline_method_id,
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
