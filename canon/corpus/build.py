from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.claims.conflict import detect_conflicts
from canon.claims.extract import extract_claims
from canon.config import load_settings
from canon.corpus.packs import build_topic_pack_manifest
from canon.embeddings.store import build_embedding_store
from canon.eval.rag import evaluate_rag
from canon.graph.build import graph_report
from canon.ingest.pipeline import run as run_ingest
from canon.quality.diagnostics import quality_diagnostics
from canon.workbench.build import build_workbench


def build_named_corpus(
    corpus_id: str,
    pack_id: str = "social_science_ir",
    max_results: int = 25,
    from_modes: list[str] | None = None,
    harvest: bool = False,
    enrich_authors: bool = True,
    resume: bool = True,
) -> dict:
    settings = load_settings()
    pack_manifest = build_topic_pack_manifest(max_results=max_results)
    source_modes = from_modes or []
    topic_runs = []
    if harvest:
        source_modes = []
        for topic in pack_manifest["topics"]:
            topic_id = topic["topic_id"]
            mode = f"{corpus_id}_{topic_id}"
            topic_file = relative_to_root(Path(topic["topic_file"]), settings.root)
            if resume and mode_files_exist(settings.data_dir, mode):
                topic_runs.append(resumed_topic_run(settings.reports_dir, mode, topic_file))
            else:
                topic_runs.append(
                    run_ingest(
                        dry_run=False,
                        live=True,
                        max_results=max_results,
                        enrich_authors=enrich_authors,
                        topic_file=topic_file,
                        output_mode=mode,
                    )
                )
            source_modes.append(mode)
    if not source_modes:
        source_modes = ["live"]
    raw_works, works, chunks, provenance = merge_modes(settings.data_dir, source_modes)
    write_json(settings.data_dir / "raw" / f"raw_{corpus_id}.json", raw_works)
    if all(raw_source_kind(raw_mode_path(settings.data_dir, mode)) == "openalex" for mode in source_modes):
        write_json(settings.data_dir / "raw" / f"openalex_{corpus_id}.json", raw_works)
    write_json(settings.data_dir / "processed" / f"works_{corpus_id}.json", works)
    write_json(settings.data_dir / "processed" / f"chunks_{corpus_id}.json", chunks)
    manifest = {
        "corpus_id": corpus_id,
        "pack_id": pack_id,
        "source_modes": source_modes,
        "harvested": harvest,
        "topic_runs": topic_runs,
        "work_count": len(works),
        "chunk_count": len(chunks),
        "provenance": provenance,
    }
    write_json(settings.reports_dir / f"corpus_{corpus_id}.json", manifest)
    return manifest


def run_phase16(
    corpus_id: str = "social_science_ir_v1",
    max_results: int = 25,
    from_modes: list[str] | None = None,
    harvest: bool = False,
    policies: list[str] | None = None,
    top_k: int = 5,
    corpus_only: bool = False,
    enrich_authors: bool = True,
    resume: bool = True,
) -> dict:
    settings = load_settings()
    corpus = build_named_corpus(
        corpus_id=corpus_id,
        max_results=max_results,
        from_modes=from_modes,
        harvest=harvest,
        enrich_authors=enrich_authors,
        resume=resume,
    )
    if corpus_only:
        report = {"corpus": corpus, "validation": {"status": "corpus_only"}}
        write_json(settings.reports_dir / f"phase16_{corpus_id}.json", report)
        return report
    quality = quality_diagnostics(corpus_id)
    graph = graph_report(corpus_id)
    claims = extract_claims(corpus_id)
    conflicts = detect_conflicts(corpus_id)
    embeddings = build_embedding_store(corpus_id, "local")
    rag = evaluate_rag(corpus_id, top_k, policies or ["lexical", "balanced", "semantic", "rag"])
    workbench = build_workbench(corpus_id)
    validation = validation_report(corpus_id, rag, claims, conflicts, quality)
    report = {
        "corpus": corpus,
        "quality": summarize_keys(quality, ["work_count", "chunk_count"]),
        "graph": {
            "node_count": graph.get("node_count"),
            "edge_count": graph.get("edge_count"),
            "component_count": len(graph.get("components", [])),
        },
        "claims": summarize_keys(claims, ["claim_count", "work_count"]),
        "conflicts": summarize_keys(conflicts, ["claim_count", "conflict_count"]),
        "embeddings": summarize_keys(embeddings, ["embedding_count", "dimensions"]),
        "rag": summarize_keys(rag, ["query_count"]),
        "validation": validation,
        "workbench": workbench,
    }
    write_json(settings.reports_dir / f"phase16_{corpus_id}.json", report)
    return report


def merge_modes(data_dir: Path, modes: list[str]) -> tuple[list[dict], list[dict], list[dict], dict]:
    raw_by_id: dict[str, dict] = {}
    works_by_id: dict[str, dict] = {}
    chunks_by_id: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    for mode in modes:
        works_path = data_dir / "processed" / f"works_{mode}.json"
        chunks_path = data_dir / "processed" / f"chunks_{mode}.json"
        raw_path = raw_mode_path(data_dir, mode)
        if not works_path.exists() or not chunks_path.exists() or not raw_path.exists():
            raise FileNotFoundError(f"Missing processed corpus files for mode '{mode}'.")
        raw_works = json.loads(raw_path.read_text(encoding="utf-8"))
        works = json.loads(works_path.read_text(encoding="utf-8"))
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        for raw_work in raw_works:
            raw_id = raw_work.get("id")
            if raw_id:
                raw_by_id.setdefault(raw_id, raw_work)
        for work in works:
            work_id = work["id"]
            current = works_by_id.setdefault(work_id, dict(work))
            modes_for_work = set(current.get("corpus_source_modes", []))
            modes_for_work.add(mode)
            current["corpus_source_modes"] = sorted(modes_for_work)
            provenance.setdefault(work_id, {"source_modes": []})
            provenance[work_id]["source_modes"] = sorted(
                set(provenance[work_id]["source_modes"]) | {mode}
            )
            provenance[work_id]["raw_source_kind"] = raw_source_kind(raw_path)
        for chunk in chunks:
            chunk_id = chunk["id"]
            current_chunk = chunks_by_id.setdefault(chunk_id, dict(chunk))
            modes_for_chunk = set(current_chunk.get("corpus_source_modes", []))
            modes_for_chunk.add(mode)
            current_chunk["corpus_source_modes"] = sorted(modes_for_chunk)
    return (
        sorted(raw_by_id.values(), key=lambda work: (work.get("publication_year") or 0, work.get("title") or "")),
        sorted(works_by_id.values(), key=lambda work: (work.get("year") or 0, work.get("title") or "")),
        sorted(chunks_by_id.values(), key=lambda chunk: chunk["id"]),
        provenance,
    )


def mode_files_exist(data_dir: Path, mode: str) -> bool:
    return all(
        [
            raw_mode_path(data_dir, mode).exists(),
            (data_dir / "processed" / f"works_{mode}.json").exists(),
            (data_dir / "processed" / f"chunks_{mode}.json").exists(),
        ]
    )


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


def raw_source_kind(path: Path) -> str:
    name = path.name
    if name.startswith("openalex_"):
        return "openalex"
    if name.startswith("unstructured_"):
        return "unstructured_jsonl"
    return "raw_json"


def resumed_topic_run(reports_dir: Path, mode: str, topic_file: str) -> dict:
    report_path = reports_dir / f"phase1_ingest_{mode}.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"mode": mode, "topic_file": topic_file}
    report["resumed"] = True
    return report


def validation_report(
    corpus_id: str,
    rag_report: dict,
    claims_report: dict,
    conflicts_report: dict,
    quality_report: dict,
) -> dict:
    policy_summary = rag_report["policy_summary"]
    return {
        "corpus_id": corpus_id,
        "practice_basis": [
            "context_relevance",
            "answer_faithfulness",
            "answer_relevance",
            "citation_support",
            "evidence_coverage",
            "human_label_loop",
        ],
        "claim_count": claims_report["claim_count"],
        "conflict_count": conflicts_report["conflict_count"],
        "quality_coverage": quality_report.get("coverage", {}),
        "policy_summary": policy_summary,
        "warnings": validation_warnings(rag_report, claims_report, conflicts_report),
    }


def validation_warnings(rag_report: dict, claims_report: dict, conflicts_report: dict) -> list[str]:
    warnings = []
    if claims_report["claim_count"] == 0:
        warnings.append("No claims were extracted; answer faithfulness cannot be claim-checked.")
    if conflicts_report["conflict_count"] == 0:
        warnings.append("No conflict candidates were found; disagreement coverage may be weak.")
    for policy, summary in rag_report["policy_summary"].items():
        if summary.get("average_claim_citation_count", 0.0) < 1.0:
            warnings.append(f"Policy '{policy}' has low claim-backed citation coverage.")
        if summary.get("average_distinct_clusters", 0.0) < 2.0:
            warnings.append(f"Policy '{policy}' has low graph-cluster diversity.")
    return warnings


def summarize_keys(payload: dict, keys: list[str]) -> dict:
    return {key: payload.get(key) for key in keys}


def cli_summary(report: dict) -> dict:
    corpus = report.get("corpus") or {}
    provenance = corpus.get("provenance") or {}
    summarized_corpus = {
        key: corpus.get(key)
        for key in [
            "corpus_id",
            "pack_id",
            "source_modes",
            "harvested",
            "work_count",
            "chunk_count",
        ]
        if key in corpus
    }
    if provenance:
        summarized_corpus["provenance_work_count"] = len(provenance)
    topic_runs = corpus.get("topic_runs") or []
    if topic_runs:
        summarized_corpus["topic_runs"] = [
            {
                "mode": run.get("mode"),
                "resumed": run.get("resumed", False),
                "work_count": run.get("work_count"),
                "chunk_count": run.get("chunk_count"),
            }
            for run in topic_runs
        ]
    summary = dict(report)
    summary["corpus"] = summarized_corpus
    return summary


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate a named CANON corpus.")
    parser.add_argument("--corpus-id", default="social_science_ir_v1")
    parser.add_argument("--max-results", type=int, default=25)
    parser.add_argument("--from-modes", default="live")
    parser.add_argument("--harvest", action="store_true")
    parser.add_argument("--corpus-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-enrich-authors", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--policies", default="lexical,balanced,semantic,rag")
    args = parser.parse_args()
    from_modes = [mode.strip() for mode in args.from_modes.split(",") if mode.strip()]
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    report = run_phase16(
        corpus_id=args.corpus_id,
        max_results=args.max_results,
        from_modes=None if args.harvest else from_modes,
        harvest=args.harvest,
        corpus_only=args.corpus_only,
        enrich_authors=not args.no_enrich_authors,
        resume=not args.no_resume,
        top_k=args.top_k,
        policies=policies,
    )
    print(json.dumps(cli_summary(report), indent=2))


if __name__ == "__main__":
    main()
