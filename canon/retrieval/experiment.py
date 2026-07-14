from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

from canon.config import load_settings
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.retrieval.clusters import load_cluster_assignments
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus
from canon.retrieval.engine import retrieve_with_stages


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@lru_cache(maxsize=8)
def cached_documents(mode: str) -> tuple[RetrievalDocument, ...]:
    settings = load_settings()
    return tuple(
        load_processed_corpus(
            settings.data_dir,
            mode,
            cluster_assignments=load_cluster_assignments(settings.data_dir, mode),
        )
    )


def run(query: str, policy: str, mode: str, top_k: int | None) -> dict:
    settings = load_settings()
    documents = cached_documents(mode)
    retrieval_settings = settings.raw["retrieval"]
    policies = retrieval_settings["policies"]
    if policy not in policies:
        raise ValueError(f"Unknown policy '{policy}'. Available: {', '.join(sorted(policies))}")
    resolved_top_k = top_k or int(retrieval_settings["top_k"])
    staged = retrieve_with_stages(
        query=query,
        documents=documents,
        weights={key: float(value) for key, value in policies[policy].items()},
        top_k=resolved_top_k,
        preview_chars=int(retrieval_settings["preview_chars"]),
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "query": query,
        "policy": policy,
        "mode": mode,
        "top_k": resolved_top_k,
        "stage_summary": staged["stage_summary"],
        "rejected_candidates": staged["rejected_candidates"],
        "results": [item.to_dict() for item in staged["trace"]],
    }
    report["contract_validation"] = validate_report(report, "retrieval")
    slug = "".join(character if character.isalnum() else "-" for character in query.lower())[:60].strip("-")
    write_json(settings.reports_dir / f"retrieval_{mode}_{policy}_{slug}.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CANON retrieval experiment.")
    parser.add_argument("query")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    settings = load_settings()
    policy = args.policy or settings.raw["retrieval"]["default_policy"]
    print(json.dumps(run(args.query, policy, args.mode, args.top_k), indent=2))


if __name__ == "__main__":
    main()
