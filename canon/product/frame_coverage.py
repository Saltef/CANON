from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.intelligence.evidence_runner import DEFAULT_AI_INFRA_FRAME, DEFAULT_EVIDENCE_REQUIREMENTS
from canon.product import service


def load_json_object(path: str | None, default: dict[str, Any]) -> dict[str, Any]:
    if not path:
        return default
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a focused research-frame coverage diagnostic.")
    parser.add_argument("question")
    parser.add_argument("--mode", default="ai_infra_geo_risk_demo")
    parser.add_argument("--policy", default="rag")
    parser.add_argument("--project-id", default="ai_infra_geo_risk")
    parser.add_argument("--request-id", default="")
    parser.add_argument("--frame-json", help="Path to a research frame JSON object.")
    parser.add_argument("--requirements-json", help="Path to an evidence requirements JSON object.")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()

    frame = load_json_object(args.frame_json, DEFAULT_AI_INFRA_FRAME)
    requirements = dict(load_json_object(args.requirements_json, DEFAULT_EVIDENCE_REQUIREMENTS))
    if args.top_k:
        requirements["top_k"] = args.top_k

    report = service.frame_coverage_report(
        {
            "request_id": args.request_id,
            "project_id": args.project_id,
            "question": args.question,
            "mode": args.mode,
            "policy": args.policy,
            "research_frame": frame,
            "evidence_requirements": requirements,
            "write_report": not args.no_write_report,
        }
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
