from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from json import JSONDecodeError
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from canon.config import load_settings
from canon.claims.extract import extract_claims, load_or_extract_claims

CONFLICTING_STANCES = {
    ("supportive", "critical"),
    ("critical", "supportive"),
    ("supportive", "mixed"),
    ("mixed", "supportive"),
}


@dataclass(frozen=True)
class ConflictCandidate:
    id: str
    left_claim_id: str
    right_claim_id: str
    left_title: str
    right_title: str
    shared_topic_tokens: list[str]
    conflict_type: str
    reason: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def shared_topic_tokens(left: dict, right: dict) -> list[str]:
    return sorted(set(left.get("topic_tokens") or []) & set(right.get("topic_tokens") or []))


def classify_conflict(left: dict, right: dict, shared: list[str]) -> tuple[str, str, float] | None:
    if len(shared) < 2:
        return None
    stance_pair = (left.get("stance"), right.get("stance"))
    if stance_pair not in CONFLICTING_STANCES:
        return None
    if left.get("cluster_id") != right.get("cluster_id"):
        return (
            "cross_cluster_tension",
            "Claims share topic language but come from different graph clusters and conflicting stances.",
            0.8,
        )
    return (
        "within_cluster_tension",
        "Claims share topic language within one graph cluster but have conflicting stances.",
        0.65,
    )


def conflict_id(left_id: str, right_id: str) -> str:
    digest = hashlib.sha1(f"{left_id}:{right_id}".encode("utf-8")).hexdigest()[:16]
    return f"conflict:{digest}"


def candidate_pair_indices(
    claims: list[dict],
    max_bucket_size: int = 500,
    max_candidates_per_claim: int = 2000,
) -> list[tuple[int, int]]:
    token_index: dict[str, list[int]] = {}
    for index, claim in enumerate(claims):
        for token in set(claim.get("topic_tokens") or []):
            token_index.setdefault(token, []).append(index)

    pairs = []
    seen = set()
    for left_index, claim in enumerate(claims):
        candidate_indices = set()
        for token in claim.get("topic_tokens") or []:
            bucket = token_index.get(token) or []
            if len(bucket) <= max_bucket_size:
                candidate_indices.update(bucket)
            if len(candidate_indices) >= max_candidates_per_claim:
                break
        for right_index in candidate_indices:
            if right_index <= left_index:
                continue
            pair = (left_index, right_index)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def detect_conflicts_from_claims(claims: list[dict]) -> list[ConflictCandidate]:
    candidates = []
    for left_index, right_index in candidate_pair_indices(claims):
        left = claims[left_index]
        right = claims[right_index]
        if left["work_id"] == right["work_id"]:
            continue
        shared = shared_topic_tokens(left, right)
        classified = classify_conflict(left, right, shared)
        if not classified:
            continue
        conflict_type, reason, score = classified
        candidates.append(
            ConflictCandidate(
                id=conflict_id(left["id"], right["id"]),
                left_claim_id=left["id"],
                right_claim_id=right["id"],
                left_title=left["title"],
                right_title=right["title"],
                shared_topic_tokens=shared,
                conflict_type=conflict_type,
                reason=reason,
                score=score,
            )
        )
    return sorted(candidates, key=lambda item: (-item.score, item.id))


def detect_conflicts(mode: str) -> dict:
    settings = load_settings()
    claims_report = load_or_extract_claims(mode)
    candidates = detect_conflicts_from_claims(claims_report["claims"])
    report = {
        "mode": mode,
        "claim_count": claims_report["claim_count"],
        "conflict_count": len(candidates),
        "conflicts": [candidate.to_dict() for candidate in candidates],
    }
    write_json(settings.data_dir / "processed" / f"conflicts_{mode}.json", report)
    write_json(settings.reports_dir / f"conflicts_{mode}.json", report)
    load_or_detect_conflicts.cache_clear()
    return report


@lru_cache(maxsize=8)
def load_or_detect_conflicts(mode: str) -> dict:
    settings = load_settings()
    conflicts_path = settings.data_dir / "processed" / f"conflicts_{mode}.json"
    if conflicts_path.exists():
        try:
            return json.loads(conflicts_path.read_text(encoding="utf-8"))
        except JSONDecodeError:
            pass
    return detect_conflicts(mode)


def cli_summary(report: dict) -> dict:
    type_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    for conflict in report.get("conflicts", []):
        conflict_type = conflict.get("conflict_type") or "unknown"
        type_counts[conflict_type] = type_counts.get(conflict_type, 0) + 1
        for token in conflict.get("shared_topic_tokens") or []:
            token_counts[token] = token_counts.get(token, 0) + 1
    top_tokens = [
        {"token": token, "count": count}
        for token, count in sorted(token_counts.items(), key=lambda row: (-row[1], row[0]))[:10]
    ]
    return {
        "mode": report["mode"],
        "claim_count": report.get("claim_count", 0),
        "conflict_count": report.get("conflict_count", 0),
        "conflict_type_counts": dict(sorted(type_counts.items())),
        "top_shared_topic_tokens": top_tokens,
        "artifacts": {
            "processed": f"data/processed/conflicts_{report['mode']}.json",
            "report": f"reports/conflicts_{report['mode']}.json",
        },
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    replace_with_retry(tmp_path, path)


def replace_with_retry(tmp_path: Path, path: Path, attempts: int = 5) -> None:
    for attempt in range(attempts):
        try:
            tmp_path.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect deterministic conflict candidates.")
    parser.add_argument("--mode", default="dry_run")
    args = parser.parse_args()
    print(json.dumps(cli_summary(detect_conflicts(args.mode)), indent=2))


if __name__ == "__main__":
    main()
