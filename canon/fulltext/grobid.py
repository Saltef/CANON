from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings


def build_grobid_plan(mode: str) -> dict:
    settings = load_settings()
    works_path = settings.data_dir / "processed" / f"works_{mode}.json"
    works = json.loads(works_path.read_text(encoding="utf-8")) if works_path.exists() else []
    candidates = [
        {
            "work_id": work["id"],
            "title": work["title"],
            "pdf_url": work.get("pdf_url"),
            "has_pdf": bool(work.get("pdf_url")),
            "target_tei_path": str(settings.data_dir / "raw" / "tei" / f"{safe_name(work['id'])}.tei.xml"),
        }
        for work in works
    ]
    report = {
        "mode": mode,
        "work_count": len(works),
        "pdf_candidate_count": sum(1 for item in candidates if item["has_pdf"]),
        "grobid_endpoint": "http://grobid:8070/api/processFulltextDocument",
        "status": "plan_only",
        "candidates": candidates,
    }
    write_json(settings.reports_dir / f"grobid_plan_{mode}.json", report)
    return report


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)[:80]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CANON GROBID full-text processing plan.")
    parser.add_argument("--mode", default="dry_run")
    args = parser.parse_args()
    print(json.dumps(build_grobid_plan(args.mode), indent=2))


if __name__ == "__main__":
    main()
