from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.product.flagship_handoff import (
    DEFAULT_FIXTURE,
    DEFAULT_MODE,
    DEFAULT_QUESTION,
    run_flagship_handoff,
)


def run_demo(
    mode: str = DEFAULT_MODE,
    question: str = DEFAULT_QUESTION,
    fixture_path: Path | None = None,
    queries_path: Path | None = None,
    policy: str = "rag",
    project_id: str = "ai_infra_geo_risk",
    write_report: bool = True,
) -> dict[str, Any]:
    handoff = run_flagship_handoff(
        mode=mode,
        question=question,
        fixture_path=fixture_path or DEFAULT_FIXTURE,
        queries_path=queries_path,
        policy=policy,
        project_id=project_id,
        write_report=write_report,
    )
    return compact_demo_summary(handoff)


def compact_demo_summary(handoff: dict[str, Any]) -> dict[str, Any]:
    review = handoff.get("human_review_boundary") or {}
    return {
        "status": handoff.get("status"),
        "mode": handoff.get("mode"),
        "policy": handoff.get("policy"),
        "question": handoff.get("question"),
        "automated_steps": compact_steps(handoff.get("steps") or []),
        "human_review": {
            "required": bool(review.get("required", True)),
            "status": review.get("status"),
            "reviewed_question_count": review.get("reviewed_question_count", 0),
            "question_count": review.get("question_count", 0),
            "next_required_action": review.get("next_required_action"),
        },
        "key_artifacts": key_artifacts(handoff.get("artifacts") or []),
        "claim": handoff.get("claim"),
        "limitations": handoff.get("limitations", []),
    }


def compact_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": step.get("id"),
            "status": step.get("status"),
            "passed": bool(step.get("passed")),
        }
        for step in steps
    ]


def key_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        "flagship_handoff_json",
        "flagship_handoff_markdown",
        "intelligence_brief_json",
        "alert_digest_json",
        "intelligence_review_tasks_json",
    }
    return [
        {
            "id": artifact.get("id"),
            "path": artifact.get("path"),
            "exists": bool(artifact.get("exists")),
            "sha256": artifact.get("sha256"),
        }
        for artifact in artifacts
        if artifact.get("id") in wanted
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CANON's built-in flagship demo and print a compact handoff summary."
    )
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--queries-path", type=Path)
    parser.add_argument("--policy", default="rag")
    parser.add_argument("--project-id", default="ai_infra_geo_risk")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    summary = run_demo(
        mode=args.mode,
        question=args.question,
        fixture_path=args.fixture,
        queries_path=args.queries_path,
        policy=args.policy,
        project_id=args.project_id,
        write_report=not args.no_write_report,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
