from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings


def judge_task(task: dict) -> dict:
    metrics = task["metrics"]
    score = 0.0
    reasons = []
    if metrics["citation_count"] >= 3:
        score += 0.25
        reasons.append("has_three_or_more_citations")
    if metrics["claim_citation_count"] >= 1:
        score += 0.25
        reasons.append("has_claim_backing")
    if metrics["distinct_clusters"] >= 2:
        score += 0.25
        reasons.append("has_cluster_diversity")
    if metrics["conflict_note_count"] >= 1:
        score += 0.25
        reasons.append("surfaces_conflict_notes")
    return {
        "task_id": task["id"],
        "policy": task["policy"],
        "heuristic_score": round(score, 6),
        "reasons": reasons,
    }


def judge_label_tasks(mode: str) -> dict:
    settings = load_settings()
    path = settings.reports_dir / f"label_tasks_{mode}.json"
    tasks_report = json.loads(path.read_text(encoding="utf-8"))
    judgments = [judge_task(task) for task in tasks_report["tasks"]]
    report = {
        "mode": mode,
        "task_count": len(judgments),
        "average_score": average(judgment["heuristic_score"] for judgment in judgments),
        "judgments": judgments,
    }
    write_json(settings.reports_dir / f"label_judgments_{mode}.json", report)
    return report


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic CANON label judgments.")
    parser.add_argument("--mode", default="dry_run")
    args = parser.parse_args()
    print(json.dumps(judge_label_tasks(args.mode), indent=2))


if __name__ == "__main__":
    main()
