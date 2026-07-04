from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from canon.models import Work


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            positioned.append((position, word))
    return " ".join(word for _, word in sorted(positioned))


def normalize_work(raw: dict[str, Any]) -> Work:
    primary_location = raw.get("primary_location") or {}
    best_oa_location = raw.get("best_oa_location") or {}
    source = (primary_location.get("source") or best_oa_location.get("source") or {})
    open_access = raw.get("open_access") or {}
    pdf_url = (
        primary_location.get("pdf_url")
        or best_oa_location.get("pdf_url")
        or open_access.get("oa_url")
    )
    author_summary = summarize_authorships(raw.get("authorships") or [])
    return Work(
        id=raw.get("id", ""),
        doi=raw.get("doi"),
        title=(raw.get("title") or "").strip(),
        year=raw.get("publication_year"),
        language=raw.get("language"),
        abstract=abstract_from_inverted_index(raw.get("abstract_inverted_index")),
        source_name=source.get("display_name"),
        is_open_access=bool(open_access.get("is_oa")),
        is_retracted=bool(raw.get("is_retracted")),
        cited_by_count=int(raw.get("cited_by_count") or 0),
        referenced_work_count=len(raw.get("referenced_works") or []),
        pdf_url=pdf_url,
        landing_page_url=primary_location.get("landing_page_url") or best_oa_location.get("landing_page_url"),
        raw=raw,
        author_display_names=author_summary["display_names"],
        author_count=author_summary["author_count"],
        max_author_cited_by_count=author_summary["max_cited_by_count"],
        max_author_works_count=author_summary["max_works_count"],
    )


def summarize_authorships(authorships: list[dict[str, Any]]) -> dict[str, Any]:
    display_names: list[str] = []
    cited_by_counts: list[int] = []
    works_counts: list[int] = []
    for authorship in authorships:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            display_names.append(name)
        cited_by_counts.append(int(author.get("cited_by_count") or 0))
        works_counts.append(int(author.get("works_count") or 0))
    return {
        "display_names": display_names,
        "author_count": len(authorships),
        "max_cited_by_count": max(cited_by_counts, default=0),
        "max_works_count": max(works_counts, default=0),
    }


def author_ids(raw_work: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for authorship in raw_work.get("authorships") or []:
        author = authorship.get("author") or {}
        author_id = author.get("id")
        if author_id:
            ids.append(author_id)
    return ids


def author_api_url(author_id: str) -> str:
    if author_id.startswith("https://api.openalex.org/authors/"):
        return author_id
    if author_id.startswith("https://openalex.org/A"):
        return "https://api.openalex.org/authors/" + author_id.rsplit("/", 1)[-1]
    return author_id


def openalex_entity_id(entity_url: str) -> str:
    return entity_url.rstrip("/").rsplit("/", 1)[-1]


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_author_batch_url(
    base_url: str,
    mailto: str,
    author_ids: list[str],
    per_page: int = 100,
    api_key: str | None = None,
) -> str:
    openalex_ids = "|".join(openalex_entity_id(author_id) for author_id in author_ids)
    params = {
        "filter": f"openalex:{openalex_ids}",
        "per-page": str(per_page),
        "mailto": mailto,
        "select": "id,display_name,cited_by_count,works_count,summary_stats",
    }
    if api_key:
        params["api_key"] = api_key
    return f"{base_url}/authors?{urllib.parse.urlencode(params)}"


def enrich_works_with_author_metrics(
    raw_works: list[dict[str, Any]],
    polite_delay_seconds: float = 0.05,
    batch_size: int = 100,
    base_url: str = "https://api.openalex.org",
    mailto: str = "replace-me@example.com",
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    unique_author_ids = sorted({author_id for raw_work in raw_works for author_id in author_ids(raw_work)})
    for batch in batched(unique_author_ids, max(1, min(batch_size, 100))):
        try:
            payload = fetch_json(
                build_author_batch_url(
                    base_url,
                    mailto,
                    batch,
                    per_page=len(batch),
                    api_key=api_key,
                )
            )
            for record in payload.get("results", []):
                record_id = record.get("id")
                if record_id:
                    cache[record_id] = record
                    cache[openalex_entity_id(record_id)] = record
            time.sleep(polite_delay_seconds)
        except Exception:
            for author_id in batch:
                try:
                    cache[author_id] = fetch_json(author_api_url(author_id))
                    time.sleep(polite_delay_seconds)
                except Exception:
                    cache[author_id] = {}

    enriched: list[dict[str, Any]] = []
    for raw_work in raw_works:
        copy = dict(raw_work)
        authorships = []
        for authorship in raw_work.get("authorships") or []:
            author = dict(authorship.get("author") or {})
            author_id = author.get("id")
            author_record = cache.get(author_id, {}) or cache.get(openalex_entity_id(author_id or ""), {})
            if author_record:
                author["cited_by_count"] = author_record.get("cited_by_count") or author.get("cited_by_count")
                author["works_count"] = author_record.get("works_count") or author.get("works_count")
            authorship_copy = dict(authorship)
            authorship_copy["author"] = author
            authorships.append(authorship_copy)
        copy["authorships"] = authorships
        enriched.append(copy)
    return enriched


def build_filter(settings: dict[str, Any], topic_ids: Iterable[str]) -> str:
    openalex = settings["openalex"]
    parts = [
        "type:article",
        f"from_publication_date:{openalex['from_publication_year']}-01-01",
        f"to_publication_date:{openalex['to_publication_year']}-12-31",
        f"language:{openalex['language']}",
    ]
    ids = "|".join(topic_ids)
    if ids:
        parts.append(f"concepts.id:{ids}")
    return ",".join(parts)


def build_url(
    base_url: str,
    mailto: str,
    filters: str,
    search: str | None,
    per_page: int,
    cursor: str = "*",
    sort: str | None = None,
    api_key: str | None = None,
) -> str:
    params = {
        "filter": filters,
        "per-page": str(per_page),
        "cursor": cursor,
        "mailto": mailto,
        "select": ",".join(
            [
                "id",
                "doi",
                "title",
                "publication_year",
                "language",
                "abstract_inverted_index",
                "primary_location",
                "best_oa_location",
                "open_access",
                "is_retracted",
                "cited_by_count",
                "referenced_works",
                "authorships",
            ]
        ),
    }
    if search:
        params["search"] = search
    if sort:
        params["sort"] = sort
    if api_key:
        params["api_key"] = api_key
    return f"{base_url}/works?{urllib.parse.urlencode(params)}"


def fetch_json(url: str, timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "CANON/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def harvest_openalex(
    settings: dict[str, Any],
    topic_ids: list[str],
    search_terms: list[str],
    max_results: int,
    polite_delay_seconds: float = 0.15,
    sort_strategies: list[str | None] | None = None,
) -> list[dict[str, Any]]:
    base_url = settings["openalex"]["base_url"]
    per_page = min(100, int(settings["openalex"]["per_page"]))
    mailto = settings["project"]["contact_email"]
    api_key = openalex_api_key(settings)
    filters = build_filter(settings, topic_ids)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    strategies = sort_strategies or [None, "cited_by_count:desc", "publication_date:desc"]

    for term in search_terms:
        for sort in strategies:
            cursor = "*"
            while len(results) < max_results:
                payload = fetch_json(build_url(base_url, mailto, filters, term, per_page, cursor, sort, api_key))
                for item in payload.get("results", []):
                    item_id = item.get("id")
                    if item_id and item_id not in seen:
                        seen.add(item_id)
                        results.append(item)
                    if len(results) >= max_results:
                        break
                next_cursor = (payload.get("meta") or {}).get("next_cursor")
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor
                time.sleep(polite_delay_seconds)
            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break
    return results


def openalex_api_key(settings: dict[str, Any]) -> str | None:
    env_name = (settings.get("openalex") or {}).get("api_key_env")
    return os.getenv(env_name) if env_name else None


def load_fixture(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))
