from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.embeddings.store import build_embedding_store


DEFAULT_PROVIDERS = ["local", "openai", "cohere"]


def parse_providers(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_PROVIDERS
    providers = [piece.strip() for piece in value.split(",") if piece.strip()]
    return providers or DEFAULT_PROVIDERS


def compare_embedding_providers(mode: str, providers: list[str] | None = None) -> dict:
    settings = load_settings()
    rows = []
    for provider in providers or DEFAULT_PROVIDERS:
        try:
            report = build_embedding_store(mode=mode, provider_name=provider)
            rows.append(
                {
                    "provider": provider,
                    "status": "ok",
                    "model": report["model"],
                    "embedding_count": report["embedding_count"],
                    "dimensions": report["dimensions"],
                    "output_path": report["output_path"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - report availability without stopping comparison.
            rows.append(
                {
                    "provider": provider,
                    "status": "unavailable",
                    "reason": str(exc),
                }
            )
    report = {
        "mode": mode,
        "providers": rows,
        "available_providers": [row["provider"] for row in rows if row["status"] == "ok"],
        "unavailable_providers": [row["provider"] for row in rows if row["status"] != "ok"],
    }
    write_json(settings.reports_dir / f"provider_compare_{mode}.json", report)
    return report


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare available CANON embedding providers.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--providers", default=None)
    args = parser.parse_args()
    print(json.dumps(compare_embedding_providers(args.mode, parse_providers(args.providers)), indent=2))


if __name__ == "__main__":
    main()
