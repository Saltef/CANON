from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path

from canon.claims.model import ClaimModel, load_claim_model, write_model_report
from canon.config import load_settings
from canon.retrieval.clusters import load_cluster_assignments
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus
from canon.retrieval.tokenize import tokenize

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

POSITIVE_MARKERS = {
    "increase",
    "increases",
    "improve",
    "improves",
    "support",
    "supports",
    "associated",
    "effective",
    "robust",
    "significant",
    "lower",
    "reduce",
    "reduces",
}

NEGATIVE_MARKERS = {
    "not",
    "rarely",
    "mixed",
    "critique",
    "critiques",
    "critics",
    "fails",
    "failure",
    "caveat",
    "weak",
    "contradict",
    "contradicts",
    "insignificant",
}

CLAIM_MARKERS = POSITIVE_MARKERS | NEGATIVE_MARKERS | {
    "find",
    "finds",
    "found",
    "show",
    "shows",
    "argue",
    "argues",
    "evidence",
    "effect",
    "effects",
    "suggest",
    "suggests",
}


@dataclass(frozen=True)
class Claim:
    id: str
    work_id: str
    chunk_id: str
    title: str
    source_name: str | None
    year: int | None
    cluster_id: int | None
    text: str
    topic_tokens: list[str]
    stance: str
    confidence: float
    model_id: str
    features: dict[str, float]
    explanation: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    pieces = SENTENCE_RE.split(normalized)
    if len(pieces) == 1:
        return [normalized]
    return [piece.strip() for piece in pieces if piece.strip()]


def stance_for_text(text: str) -> str:
    return load_claim_model().predict(text).stance


def claim_confidence(text: str) -> float:
    return load_claim_model().predict(text).confidence


def topic_tokens_for_text(text: str, limit: int = 8) -> list[str]:
    tokens = tokenize(text)
    counts: dict[str, int] = {}
    for token in tokens:
        if len(token) < 4:
            continue
        counts[token] = counts.get(token, 0) + 1
    return [
        token
        for token, _count in sorted(counts.items(), key=lambda row: (-row[1], row[0]))[:limit]
    ]


def extract_claims_from_document(
    document: RetrievalDocument,
    model: ClaimModel | None = None,
) -> list[Claim]:
    claim_model = model or load_claim_model()
    claims = []
    for index, sentence in enumerate(split_sentences(document.text)):
        prediction = claim_model.predict(sentence)
        if not prediction.is_claim:
            continue
        claim_id = stable_claim_id(document.chunk_id, index, sentence)
        claims.append(
            Claim(
                id=claim_id,
                work_id=document.work_id,
                chunk_id=document.chunk_id,
                title=document.title,
                source_name=document.source_name,
                year=document.year,
                cluster_id=document.cluster_id,
                text=sentence,
                topic_tokens=topic_tokens_for_text(sentence),
                stance=prediction.stance,
                confidence=prediction.confidence,
                model_id=prediction.model_id,
                features=prediction.features,
                explanation=prediction.explanation,
            )
        )
    return claims


def stable_claim_id(chunk_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{chunk_id}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"claim:{digest}"


def extract_claims(mode: str) -> dict:
    settings = load_settings()
    model = load_claim_model()
    cluster_assignments = load_cluster_assignments(settings.data_dir, mode)
    documents = load_processed_corpus(settings.data_dir, mode, cluster_assignments)
    candidate_sentences = [
        sentence
        for document in documents
        for sentence in split_sentences(document.text)
    ]
    claims = [
        claim
        for document in documents
        for claim in extract_claims_from_document(document, model)
    ]
    report = {
        "mode": mode,
        "model_id": model.model_id,
        "claim_count": len(claims),
        "work_count": len({claim.work_id for claim in claims}),
        "claims": [claim.to_dict() for claim in claims],
    }
    write_json(settings.data_dir / "processed" / f"claims_{mode}.json", report)
    write_json(settings.reports_dir / f"claims_{mode}.json", report)
    write_model_report(mode, candidate_sentences)
    load_or_extract_claims.cache_clear()
    return report


@lru_cache(maxsize=8)
def load_or_extract_claims(mode: str) -> dict:
    settings = load_settings()
    claims_path = settings.data_dir / "processed" / f"claims_{mode}.json"
    if claims_path.exists():
        try:
            return json.loads(claims_path.read_text(encoding="utf-8"))
        except JSONDecodeError:
            pass
    return extract_claims(mode)


def cli_summary(report: dict) -> dict:
    return {
        "mode": report["mode"],
        "model_id": report.get("model_id"),
        "claim_count": report.get("claim_count", 0),
        "work_count": report.get("work_count", 0),
        "sample_claim_ids": [claim["id"] for claim in report.get("claims", [])[:5]],
        "artifacts": {
            "processed": f"data/processed/claims_{report['mode']}.json",
            "report": f"reports/claims_{report['mode']}.json",
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
    parser = argparse.ArgumentParser(description="Extract deterministic claim candidates.")
    parser.add_argument("--mode", default="dry_run")
    args = parser.parse_args()
    print(json.dumps(cli_summary(extract_claims(args.mode)), indent=2))


if __name__ == "__main__":
    main()
