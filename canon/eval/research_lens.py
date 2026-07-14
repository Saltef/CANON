from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product.service import build_query_diagnostics
from canon.synthesis.answer import synthesize


DEFAULT_LENSES = {
    "broad_discovery": {
        "label": "Broad discovery",
        "policy": "focus_diverse",
        "freedom_level": "exploratory",
        "intent": "Map evidence neighborhoods, related terminology, and source breadth before making claims.",
    },
    "canonical_depth": {
        "label": "Canonical depth",
        "policy": "source_quality_heavy",
        "freedom_level": "balanced",
        "intent": "Prioritize stronger, more central sources for closer human inspection.",
    },
    "recent_scan": {
        "label": "Recent scan",
        "policy": "recency_heavy",
        "freedom_level": "balanced",
        "intent": "Check whether newer evidence changes the research trail.",
    },
}


def evaluate_research_lenses(
    query: str,
    mode: str,
    top_k: int = 10,
    lenses: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    lens_definitions = lenses or DEFAULT_LENSES
    rows = [
        lens_row(lens_id=lens_id, definition=definition, query=query, mode=mode, top_k=top_k)
        for lens_id, definition in lens_definitions.items()
    ]
    report = {
        "report_id": "research_lens_calibration_v1",
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "status": report_status(rows),
        "positioning": "Researcher-calibrated discovery: breadth first, with inspectable depth and human review boundaries.",
        "lenses": rows,
        "comparison": compare_lenses(rows),
        "human_review_prompts": human_review_prompts(rows),
        "limitations": [
            "Lens comparison shows retrieval behavior, not ground truth.",
            "Source and cluster diversity can indicate breadth but cannot prove comprehensive coverage.",
            "Terminology suggestions should be accepted, edited, or rejected by the researcher before they become calibration patterns.",
        ],
    }
    write_artifacts(report)
    return report


def lens_row(
    lens_id: str,
    definition: dict[str, Any],
    query: str,
    mode: str,
    top_k: int,
) -> dict[str, Any]:
    policy = str(definition["policy"])
    freedom_level = str(definition.get("freedom_level") or "balanced")
    answer = synthesize(query=query, policy=policy, mode=mode, top_k=top_k)
    diagnostics = build_query_diagnostics(
        query=query,
        mode=mode,
        top_k=min(top_k, 5),
        candidate_k=max(20, top_k * 2),
        freedom_level=freedom_level,
    )
    evidence = answer.get("evidence") or []
    return {
        "lens_id": lens_id,
        "label": definition.get("label") or lens_id.replace("_", " ").title(),
        "intent": definition.get("intent") or "",
        "policy": policy,
        "freedom_level": freedom_level,
        "answer_summary": answer.get("answer"),
        "support_assessment": answer.get("support_assessment", {}),
        "source_profile": source_profile(evidence),
        "top_evidence": compact_evidence(evidence),
        "top_score_contributors": top_score_contributors(evidence),
        "query_diagnostics": {
            "matched_terms": diagnostics["query_to_corpus"]["matched_terms"],
            "weak_terms": diagnostics["query_to_corpus"]["weak_terms"],
            "field_phrases": diagnostics["result_neighborhood"]["field_phrases"],
            "query_variants": diagnostics["query_variants"],
            "stability": diagnostics["stability"],
        },
        "researcher_actions": researcher_actions(evidence, diagnostics, lens_id),
    }


def source_profile(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    sources = Counter(source_key(item) for item in evidence)
    works = Counter(str(item.get("work_id") or "unknown") for item in evidence)
    clusters = Counter(str(item.get("cluster_id")) for item in evidence if item.get("cluster_id") is not None)
    dominant_source, dominant_count = sources.most_common(1)[0] if sources else ("none", 0)
    return {
        "evidence_count": len(evidence),
        "distinct_sources": len(sources),
        "distinct_works": len(works),
        "distinct_clusters": len(clusters),
        "dominant_source": dominant_source,
        "dominant_source_share": safe_ratio(dominant_count, len(evidence)),
        "source_counts": dict(sorted(sources.items())),
        "cluster_counts": dict(sorted(clusters.items())),
    }


def source_key(item: dict[str, Any]) -> str:
    return str(item.get("source_name") or item.get("title") or item.get("work_id") or "unknown").strip() or "unknown"


def compact_evidence(evidence: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for item in evidence[:limit]:
        rows.append(
            {
                "rank": item.get("rank"),
                "citation_id": item.get("citation_id"),
                "title": item.get("title"),
                "source_name": item.get("source_name"),
                "year": item.get("year"),
                "cluster_id": item.get("cluster_id"),
                "final_score": item.get("final_score"),
                "top_contributors": (item.get("explanation") or {}).get("top_contributors", [])[:3],
            }
        )
    return rows


def top_score_contributors(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in evidence:
        for contributor in (item.get("explanation") or {}).get("top_contributors", []):
            component = str(contributor.get("component") or "")
            if not component:
                continue
            totals[component] = totals.get(component, 0.0) + float(contributor.get("score_share") or 0.0)
            counts[component] = counts.get(component, 0) + 1
    rows = [
        {
            "component": component,
            "average_score_share": round(totals[component] / counts[component], 6),
            "evidence_mentions": counts[component],
        }
        for component in totals
    ]
    return sorted(rows, key=lambda row: (row["average_score_share"], row["evidence_mentions"]), reverse=True)


def researcher_actions(evidence: list[dict[str, Any]], diagnostics: dict[str, Any], lens_id: str) -> list[str]:
    actions = []
    profile = source_profile(evidence)
    if profile["distinct_clusters"] >= 3 and lens_id == "broad_discovery":
        actions.append("Use this lens to choose evidence neighborhoods for deeper review.")
    if profile["dominant_source_share"] > 0.4:
        actions.append("Inspect source concentration before trusting this lens.")
    if diagnostics["query_to_corpus"]["weak_terms"]:
        actions.append("Review weak query terms and replace vague wording with accepted field phrases.")
    if diagnostics["result_neighborhood"]["field_phrases"]:
        actions.append("Accept, edit, or reject suggested field phrases as researcher calibration patterns.")
    if not actions:
        actions.append("Review top evidence and mark whether the lens behavior matches the research goal.")
    return actions


def compare_lenses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    evidence_by_lens = {
        row["lens_id"]: {item["title"] for item in row["top_evidence"] if item.get("title")}
        for row in rows
    }
    pairwise_overlap = []
    lens_ids = list(evidence_by_lens)
    for index, left_id in enumerate(lens_ids):
        for right_id in lens_ids[index + 1 :]:
            left = evidence_by_lens[left_id]
            right = evidence_by_lens[right_id]
            pairwise_overlap.append(
                {
                    "left_lens": left_id,
                    "right_lens": right_id,
                    "top_evidence_overlap": safe_ratio(len(left & right), len(left | right)),
                }
            )
    return {
        "most_broad_lens": max(rows, key=lambda row: row["source_profile"]["distinct_clusters"])["lens_id"],
        "most_source_concentrated_lens": max(rows, key=lambda row: row["source_profile"]["dominant_source_share"])[
            "lens_id"
        ],
        "pairwise_overlap": pairwise_overlap,
    }


def human_review_prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompts = []
    for row in rows:
        prompts.append(
            {
                "lens_id": row["lens_id"],
                "questions": [
                    "Did this lens retrieve evidence appropriate for its stated intent?",
                    "Which suggested field phrases should be accepted, edited, or rejected?",
                    "Which source or cluster should be inspected before using this answer?",
                ],
            }
        )
    return prompts


def report_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "review"
    has_breadth = any(row["source_profile"]["distinct_clusters"] >= 3 for row in rows)
    has_depth = any(
        any(item["component"] in {"source_quality", "citation_centrality"} for item in row["top_score_contributors"])
        for row in rows
    )
    return "ready_for_human_calibration" if has_breadth and has_depth else "review"


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def write_artifacts(report: dict[str, Any]) -> None:
    settings = load_settings()
    slug = "".join(character if character.isalnum() else "-" for character in report["query"].lower())[:60].strip("-")
    json_path = settings.reports_dir / f"research_lens_{report['mode']}_{slug}.json"
    md_path = settings.reports_dir / f"research_lens_{report['mode']}_{slug}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CANON Research Lens Calibration",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Query: `{report['query']}`",
        "",
        f"Mode: `{report['mode']}`",
        "",
        "## Lens Results",
        "",
    ]
    for row in report["lenses"]:
        profile = row["source_profile"]
        phrases = ", ".join(row["query_diagnostics"]["field_phrases"][:5]) or "none"
        actions = "\n".join(f"- {action}" for action in row["researcher_actions"])
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"- Policy: `{row['policy']}`",
                f"- Intent: {row['intent']}",
                f"- Distinct sources: {profile['distinct_sources']}",
                f"- Distinct works: {profile['distinct_works']}",
                f"- Distinct clusters: {profile['distinct_clusters']}",
                f"- Dominant source share: {profile['dominant_source_share']}",
                f"- Suggested field phrases: {phrases}",
                "",
                "Researcher actions:",
                "",
                actions,
                "",
            ]
        )
    return "\n".join(lines)


def cli_summary(report: dict[str, Any]) -> dict[str, Any]:
    slug = "".join(character if character.isalnum() else "-" for character in report["query"].lower())[:60].strip("-")
    return {
        "report_id": report["report_id"],
        "status": report["status"],
        "mode": report["mode"],
        "query": report["query"],
        "lens_count": len(report["lenses"]),
        "comparison": report["comparison"],
        "artifacts": {
            "json": f"reports/research_lens_{report['mode']}_{slug}.json",
            "md": f"reports/research_lens_{report['mode']}_{slug}.md",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare researcher-calibrated discovery lenses for one query.")
    parser.add_argument("query")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    report = evaluate_research_lenses(query=args.query, mode=args.mode, top_k=args.top_k)
    print(json.dumps(cli_summary(report), indent=2))


if __name__ == "__main__":
    main()
