from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import report_io


DEFAULT_OPTIMIZER_REPORT = Path("reports") / "stage1_optimizer_v1.json"


def run_stage1_gate(
    optimizer_report: Path | None = None,
    min_queries: int = 30,
    min_corpora: int = 2,
    min_candidate_recall: float = 0.95,
    min_recall_at_k: float = 0.8,
    min_ndcg_at_k: float = 0.75,
    require_no_unavailable: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    report_path = resolve_path(settings.root, optimizer_report or DEFAULT_OPTIMIZER_REPORT)
    optimizer = json.loads(report_path.read_text(encoding="utf-8"))
    trials = optimizer.get("trials") or []
    ok_trials = [trial for trial in trials if trial.get("status") == "ok"]
    best = optimizer.get("best_trial") or first_or_empty(optimizer.get("leaderboard") or [])
    corpora = sorted({str(trial.get("benchmark_id")) for trial in trials if trial.get("benchmark_id")})
    total_queries = sum_queries_by_benchmark(trials)
    failures = gate_failures(
        trials=trials,
        ok_trials=ok_trials,
        best=best,
        total_queries=total_queries,
        corpus_count=len(corpora),
        min_queries=min_queries,
        min_corpora=min_corpora,
        min_candidate_recall=min_candidate_recall,
        min_recall_at_k=min_recall_at_k,
        min_ndcg_at_k=min_ndcg_at_k,
        require_no_unavailable=require_no_unavailable,
    )
    gate = {
        "report_id": "stage1_gate_v1",
        "status": "pass" if not failures else "blocked",
        "optimizer_report": relative(settings.root, report_path),
        "optimizer_status": optimizer.get("status"),
        "requirements": {
            "min_queries": min_queries,
            "min_corpora": min_corpora,
            "min_candidate_recall": min_candidate_recall,
            "min_recall_at_k": min_recall_at_k,
            "min_ndcg_at_k": min_ndcg_at_k,
            "require_no_unavailable": require_no_unavailable,
        },
        "observed": {
            "query_count": total_queries,
            "corpus_count": len(corpora),
            "corpora": corpora,
            "trial_count": len(trials),
            "ok_trial_count": len(ok_trials),
            "unavailable_trial_count": len(trials) - len(ok_trials),
            "best_trial": best,
        },
        "failures": failures,
        "next_steps": next_steps(failures),
    }
    if write_report:
        report_io.write_json(settings.reports_dir / "stage1_gate_v1.json", gate)
        report_io.write_markdown(settings.reports_dir / "stage1_gate_v1.md", render_markdown(gate))
    return gate


def gate_failures(
    *,
    trials: list[dict[str, Any]],
    ok_trials: list[dict[str, Any]],
    best: dict[str, Any],
    total_queries: int,
    corpus_count: int,
    min_queries: int,
    min_corpora: int,
    min_candidate_recall: float,
    min_recall_at_k: float,
    min_ndcg_at_k: float,
    require_no_unavailable: bool,
) -> list[dict[str, Any]]:
    failures = []
    if not trials:
        failures.append({"id": "no_trials", "message": "Run stage1_optimizer before the gate."})
    if total_queries < min_queries:
        failures.append(
            {
                "id": "query_scale",
                "message": f"Need at least {min_queries} evaluated public-qrels queries.",
                "observed": total_queries,
            }
        )
    if corpus_count < min_corpora:
        failures.append(
            {
                "id": "corpus_scale",
                "message": f"Need at least {min_corpora} public corpora/topics.",
                "observed": corpus_count,
            }
        )
    if not ok_trials:
        failures.append({"id": "no_ok_trials", "message": "No model stack completed successfully."})
    if require_no_unavailable and len(ok_trials) != len(trials):
        failures.append(
            {
                "id": "model_matrix_incomplete",
                "message": "At least one configured hosted model/reranker was unavailable.",
                "observed": len(trials) - len(ok_trials),
            }
        )
    if best:
        metric_checks = [
            ("candidate_recall_mean", min_candidate_recall),
            ("Recall@k", min_recall_at_k),
            ("nDCG@k", min_ndcg_at_k),
        ]
        for metric, threshold in metric_checks:
            observed = float(best.get(metric, 0.0))
            if observed < threshold:
                failures.append(
                    {
                        "id": f"{metric}_below_threshold",
                        "message": f"Best trial must reach {metric} >= {threshold}.",
                        "observed": observed,
                    }
                )
    return failures


def next_steps(failures: list[dict[str, Any]]) -> list[str]:
    lookup = {failure["id"] for failure in failures}
    steps = []
    if "query_scale" in lookup:
        steps.append("Import a 30-query public qrels set and rerun the optimizer.")
    if "corpus_scale" in lookup:
        steps.append("Add a second public corpus such as NFCorpus and include it in the suite.")
    if any("below_threshold" in item for item in lookup):
        steps.append("Expand candidate_k, fusion methods, query variants, and model matrix.")
    if "model_matrix_incomplete" in lookup:
        steps.append("Confirm API keys and provider model ids, then rerun with --no-resume if needed.")
    return steps or ["Freeze the Stage 1 report and move to human review/brief-level evaluation."]


def sum_queries_by_benchmark(trials: list[dict[str, Any]]) -> int:
    by_benchmark: dict[str, int] = {}
    for trial in trials:
        benchmark_id = str(trial.get("benchmark_id"))
        by_benchmark[benchmark_id] = max(by_benchmark.get(benchmark_id, 0), int(trial.get("query_count", 0)))
    return sum(by_benchmark.values())


def render_markdown(gate: dict[str, Any]) -> str:
    failures = "\n".join(f"- `{item['id']}`: {item['message']}" for item in gate["failures"]) or "- none"
    next_steps_rows = "\n".join(f"- {item}" for item in gate["next_steps"])
    return f"""# CANON Stage 1 Gate

Status: **{gate['status']}**

Optimizer report: `{gate['optimizer_report']}`

Observed queries: `{gate['observed']['query_count']}`

Observed corpora: `{gate['observed']['corpus_count']}`

## Failures

{failures}

## Next Steps

{next_steps_rows}
"""


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def first_or_empty(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether automated Stage 1 completion criteria pass.")
    parser.add_argument("--optimizer-report", default=str(DEFAULT_OPTIMIZER_REPORT))
    parser.add_argument("--min-queries", type=int, default=30)
    parser.add_argument("--min-corpora", type=int, default=2)
    parser.add_argument("--min-candidate-recall", type=float, default=0.95)
    parser.add_argument("--min-recall-at-k", type=float, default=0.8)
    parser.add_argument("--min-ndcg-at-k", type=float, default=0.75)
    parser.add_argument("--require-no-unavailable", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    gate = run_stage1_gate(
        optimizer_report=Path(args.optimizer_report),
        min_queries=args.min_queries,
        min_corpora=args.min_corpora,
        min_candidate_recall=args.min_candidate_recall,
        min_recall_at_k=args.min_recall_at_k,
        min_ndcg_at_k=args.min_ndcg_at_k,
        require_no_unavailable=args.require_no_unavailable,
    )
    print(json.dumps(gate, indent=2, ensure_ascii=True))
    if gate["status"] != "pass" and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
