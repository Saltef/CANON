from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from canon.config import load_settings


def load_raw_works(data_dir: Path, mode: str) -> list[dict[str, Any]]:
    path = raw_mode_path(data_dir, mode)
    return json.loads(path.read_text(encoding="utf-8"))


def raw_mode_path(data_dir: Path, mode: str) -> Path:
    candidates = [
        data_dir / "raw" / f"openalex_{mode}.json",
        data_dir / "raw" / f"unstructured_{mode}.json",
        data_dir / "raw" / f"raw_{mode}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_edges(raw_works: list[dict[str, Any]]) -> list[tuple[str, str]]:
    in_corpus = {work["id"] for work in raw_works}
    edges = []
    for work in raw_works:
        source = work["id"]
        for target in work.get("referenced_works") or []:
            if target in in_corpus:
                edges.append((source, target))
    return edges


def matched_reference_fraction(raw_works: list[dict[str, Any]]) -> float:
    in_corpus = {work["id"] for work in raw_works}
    total = 0
    matched = 0
    for work in raw_works:
        refs = work.get("referenced_works") or []
        total += len(refs)
        matched += sum(1 for ref in refs if ref in in_corpus)
    if total == 0:
        return 0.0
    return round(matched / total, 6)


def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    seen: set[str] = set()
    components = []
    for node in sorted(nodes):
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def graph_report(mode: str) -> dict:
    settings = load_settings()
    raw_works = load_raw_works(settings.data_dir, mode)
    by_id = {work["id"]: work for work in raw_works}
    edges = build_edges(raw_works)
    degrees = Counter()
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    components = connected_components(list(by_id), edges)
    clusters = []
    for index, component in enumerate(components, start=1):
        works = [by_id[node] for node in component]
        clusters.append(
            {
                "cluster_id": index,
                "size": len(component),
                "exemplar_titles": [work.get("title") for work in works[:5]],
                "top_sources": dict(
                    Counter(
                        ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
                        or work.get("source_name")
                        or work.get("source")
                        or "UNKNOWN"
                        for work in works
                    ).most_common(5)
                ),
                "years": dict(Counter(str(work.get("publication_year") or work.get("year")) for work in works)),
            }
        )
    report = {
        "mode": mode,
        "node_count": len(by_id),
        "edge_count": len(edges),
        "matched_reference_fraction": matched_reference_fraction(raw_works),
        "components": clusters,
        "degree": {
            node: degrees[node]
            for node in sorted(by_id, key=lambda node: (-degrees[node], node))
        },
    }
    write_json(settings.reports_dir / f"graph_{mode}.json", report)
    return report


def cli_summary(report: dict) -> dict:
    components = report.get("components", [])
    largest_components = sorted(components, key=lambda row: row["size"], reverse=True)[:10]
    degree = report.get("degree", {})
    top_degree = [
        {"work_id": work_id, "degree": value}
        for work_id, value in sorted(degree.items(), key=lambda row: (-row[1], row[0]))[:10]
    ]
    return {
        "mode": report["mode"],
        "node_count": report.get("node_count", 0),
        "edge_count": report.get("edge_count", 0),
        "matched_reference_fraction": report.get("matched_reference_fraction", 0.0),
        "component_count": len(components),
        "largest_components": largest_components,
        "top_degree": top_degree,
        "artifacts": {
            "report": f"reports/graph_{report['mode']}.json",
        },
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build citation graph diagnostics.")
    parser.add_argument("--mode", default="dry_run")
    args = parser.parse_args()
    print(json.dumps(cli_summary(graph_report(args.mode)), indent=2))


if __name__ == "__main__":
    main()
