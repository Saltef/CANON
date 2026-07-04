from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from canon.config import load_settings


def build_topic_pack_manifest(pack_path: Path | None = None, max_results: int = 50) -> dict:
    settings = load_settings()
    resolved_pack_path = pack_path or settings.root / "conf" / "topic_packs" / "social_science_ir.json"
    pack = json.loads(resolved_pack_path.read_text(encoding="utf-8"))
    topics = []
    for topic_path in pack["topics"]:
        topic_file = settings.root / topic_path
        topic = tomllib.loads(topic_file.read_text(encoding="utf-8"))
        topics.append(topic_manifest_row(topic, topic_file, topic_path, max_results))
    manifest = {
        "pack_id": pack["pack_id"],
        "name": pack["name"],
        "topic_count": len(topics),
        "max_results_per_topic": max_results,
        "topics": topics,
    }
    write_json(settings.reports_dir / f"topic_pack_{pack['pack_id']}.json", manifest)
    return manifest


def topic_manifest_row(topic: dict, topic_file: Path, topic_path: str, max_results: int) -> dict:
    if "topic" in topic:
        topic_id = topic["topic"]["id"]
        name = topic["topic"]["name"]
        query = topic["topic"]["query"]
    else:
        scope = topic["scope"]
        topic_id = topic_file.stem
        name = scope["label"]
        query = "; ".join(scope.get("seed_terms", [])[:5])
    return {
        "topic_id": topic_id,
        "name": name,
        "query": query,
        "topic_file": str(topic_file),
        "recommended_command": (
            "python -m canon.ingest.pipeline --live "
            f"--max-results {max_results} --topic-file {topic_path}"
        ),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CANON topic-pack manifest.")
    parser.add_argument("--pack", default=None)
    parser.add_argument("--max-results", type=int, default=50)
    args = parser.parse_args()
    pack_path = Path(args.pack) if args.pack else None
    print(json.dumps(build_topic_pack_manifest(pack_path, args.max_results), indent=2))


if __name__ == "__main__":
    main()
