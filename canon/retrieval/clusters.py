from __future__ import annotations

from pathlib import Path

from canon.graph.build import build_edges, connected_components, load_raw_works


def load_cluster_assignments(data_dir: Path, mode: str) -> dict[str, int]:
    raw_works = load_raw_works(data_dir, mode)
    nodes = [work["id"] for work in raw_works]
    edges = build_edges(raw_works)
    components = connected_components(nodes, edges)
    assignments: dict[str, int] = {}
    for cluster_id, component in enumerate(components, start=1):
        for work_id in component:
            assignments[work_id] = cluster_id
    return assignments
