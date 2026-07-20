from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from canon.config import load_settings
from canon.product import report_io
from canon.retrieval.clusters import load_cluster_assignments
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus
from canon.retrieval.tokenize import tokenize


STOP_TERMS = {
    "about",
    "after",
    "aged",
    "also",
    "all",
    "among",
    "analysis",
    "and",
    "are",
    "associated",
    "association",
    "because",
    "been",
    "between",
    "both",
    "but",
    "based",
    "can",
    "compared",
    "copyright",
    "does",
    "during",
    "elsevier",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "its",
    "less",
    "may",
    "more",
    "not",
    "observed",
    "other",
    "out",
    "over",
    "related",
    "reserved",
    "results",
    "rights",
    "significant",
    "significantly",
    "study",
    "studies",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "than",
    "this",
    "those",
    "through",
    "title",
    "under",
    "use",
    "used",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "years",
}


def topic_profile_path(data_dir: Path, mode: str) -> Path:
    return data_dir / "processed" / f"topic_profile_{mode}.json"


def load_topic_profile(data_dir: Path, mode: str) -> dict[str, Any]:
    path = topic_profile_path(data_dir, mode)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def ensure_topic_profile(data_dir: Path, mode: str, documents: Sequence[RetrievalDocument]) -> dict[str, Any]:
    loaded = load_topic_profile(data_dir, mode)
    if loaded:
        return loaded
    profile = build_topic_profile(mode=mode, documents=documents)
    write_topic_profile(data_dir, mode, profile)
    return profile


def build_topic_profile(
    mode: str,
    documents: Sequence[RetrievalDocument],
    top_terms: int = 80,
    top_phrases: int = 80,
    top_topics: int = 12,
) -> dict[str, Any]:
    term_counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    cooccurrence: dict[str, Counter[str]] = defaultdict(Counter)
    topic_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for document in documents:
        terms = content_terms(" ".join([document.title or "", document.text or ""]))
        unique_terms = set(terms)
        term_counts.update(terms)
        document_frequency.update(unique_terms)
        for left in unique_terms:
            for right in unique_terms:
                if left != right:
                    cooccurrence[left][right] += 1
        phrases = candidate_phrases(terms)
        phrase_counts.update(phrases)
        bucket = str(document.cluster_id if document.cluster_id is not None else document.work_signals.get("domain") or "corpus")
        topic_buckets[bucket].update(terms)
    doc_count = max(1, len(documents))
    scored_terms = [
        {
            "term": term,
            "score": round(count * math.log(1.0 + doc_count / max(1, document_frequency[term])), 6),
            "count": int(count),
            "document_frequency": int(document_frequency[term]),
        }
        for term, count in term_counts.items()
        if len(term) >= 3
    ]
    scored_terms.sort(key=lambda row: (row["score"], row["document_frequency"], row["term"]), reverse=True)
    scored_phrases = [
        {"phrase": phrase, "count": int(count)}
        for phrase, count in phrase_counts.most_common(top_phrases)
        if count >= 2 and len(set(phrase.split())) >= 2
    ]
    topics = [
        {
            "topic_id": topic_id,
            "terms": [term for term, _count in counts.most_common(10)],
            "document_hint": "cluster" if topic_id.isdigit() else "domain",
        }
        for topic_id, counts in sorted(topic_buckets.items(), key=lambda item: (sum(item[1].values()), item[0]), reverse=True)
    ][:top_topics]
    return {
        "report_id": "corpus_topic_profile_v1",
        "mode": mode,
        "document_count": len(documents),
        "top_terms": scored_terms[:top_terms],
        "top_phrases": scored_phrases,
        "topics": topics,
        "related_terms": related_terms(scored_terms[:top_terms], cooccurrence),
        "limitations": [
            "Topic profiles are deterministic lexical hints, not a statistical topic model.",
            "Use topic-derived variants as recall aids and keep drift review visible.",
        ],
    }


def write_topic_profile(data_dir: Path, mode: str, profile: dict[str, Any]) -> Path:
    path = topic_profile_path(data_dir, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def related_terms(scored_terms: list[dict[str, Any]], cooccurrence: dict[str, Counter[str]]) -> dict[str, list[str]]:
    vocabulary = {row["term"] for row in scored_terms}
    term_scores = {str(row["term"]): float(row["score"]) for row in scored_terms}
    related = {}
    for term in vocabulary:
        candidates = [
            item
            for item in cooccurrence.get(term, Counter())
            if item in vocabulary and item != term
        ]
        candidates.sort(key=lambda item: (cooccurrence[term][item], term_scores.get(item, 0.0), item), reverse=True)
        related[term] = [
            item
            for item in candidates[:8]
        ]
    return dict(sorted(related.items()))


def topic_query_variants(query: str, profile: dict[str, Any], max_variants: int = 4) -> list[str]:
    query_terms = set(content_terms(query))
    if not query_terms or not profile:
        return []
    related = profile.get("related_terms") or {}
    variants = []
    for term in sorted(query_terms):
        for related_term in related.get(term, [])[:3]:
            if related_term in query_terms:
                continue
            variants.append(" ".join(sorted(query_terms | {related_term})))
    for topic in profile.get("topics") or []:
        topic_terms = [str(term) for term in topic.get("terms") or []]
        overlap = query_terms & set(topic_terms)
        if overlap:
            variants.append(" ".join(sorted(query_terms | set(topic_terms[:3]))))
    for phrase in profile.get("top_phrases") or []:
        value = str(phrase.get("phrase") or "")
        phrase_terms = set(content_terms(value))
        if phrase_terms & query_terms:
            variants.append(" ".join(sorted(query_terms | phrase_terms)))
    return dedupe(variants, original=query)[:max_variants]


def content_terms(text: str) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if len(token) >= 3 and token not in STOP_TERMS and not token.isdigit()
    ]


def candidate_phrases(terms: list[str]) -> list[str]:
    phrases = []
    for size in (2, 3):
        for index in range(0, max(0, len(terms) - size + 1)):
            phrase_terms = terms[index : index + size]
            if len(set(phrase_terms)) != len(phrase_terms):
                continue
            phrases.append(" ".join(phrase_terms))
    return phrases


def dedupe(values: list[str], original: str = "") -> list[str]:
    original_key = normalize(original)
    seen = {original_key} if original_key else set()
    rows = []
    for value in values:
        key = normalize(value)
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(value)
    return rows


def normalize(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def build_topic_profile_for_mode(mode: str, write_report: bool = True) -> dict[str, Any]:
    settings = load_settings()
    documents = load_processed_corpus(
        settings.data_dir,
        mode,
        cluster_assignments=load_cluster_assignments(settings.data_dir, mode),
    )
    profile = build_topic_profile(mode=mode, documents=documents)
    path = write_topic_profile(settings.data_dir, mode, profile)
    if write_report:
        report_io.write_json(settings.reports_dir / f"topic_profile_{mode}.json", profile)
    return {**profile, "path": str(path).replace("\\", "/")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic corpus topic profile.")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_topic_profile_for_mode(args.mode, write_report=not args.no_report), indent=2))


if __name__ == "__main__":
    main()
