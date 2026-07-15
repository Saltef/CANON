from __future__ import annotations

import argparse
import json

from canon.product import service


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a focused intelligence report quality gate.")
    parser.add_argument("question")
    parser.add_argument("--mode", default="ai_infra_geo_risk_demo")
    parser.add_argument("--policy", default="rag")
    parser.add_argument("--project-id", default="ai_infra_geo_risk")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    report = service.report_quality_gate(
        {
            "question": args.question,
            "mode": args.mode,
            "policy": args.policy,
            "project_id": args.project_id,
            "write_report": not args.no_write_report,
        }
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
