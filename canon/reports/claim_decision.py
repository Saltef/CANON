from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.eval.methods import load_method_set


RESOLVABLE_WARNING_IDS = [
    "unstable_top_method",
    "narrow_top2_margin",
    "internal_qrels_only",
    "low_bootstrap_winner_probability",
    "dominated_methods",
]


def build_claim_decision(mode: str, method_set_id: str = "baseline_methods_v1") -> dict:
    settings = load_settings()
    artifacts = load_artifacts(settings.reports_dir, mode, method_set_id)
    method_set = load_method_set(None)
    role_by_method = {method["id"]: method.get("role", "candidate") for method in method_set["methods"]}
    resolutions = warning_resolutions(artifacts, role_by_method)
    report = {
        "mode": mode,
        "method_set_id": method_set_id,
        "status": "pass",
        "global_winner_claim": global_winner_claim(artifacts, resolutions),
        "benchmark_claim": benchmark_claim(artifacts, resolutions),
        "method_roles": role_by_method,
        "resolved_warning_ids": sorted(resolutions),
        "resolutions": resolutions,
    }
    stem = f"claim_decision_{mode}_{method_set_id}"
    write_json(settings.reports_dir / f"{stem}.json", report)
    write_markdown(settings.reports_dir / f"{stem}.md", report)
    return report


def load_artifacts(reports_dir: Path, mode: str, method_set_id: str) -> dict:
    patterns = {
        "batch_eval": f"batch_eval_{mode}_{method_set_id}.json",
        "evaluation_suite": f"evaluation_suite_{mode}_{method_set_id}.json",
        "bootstrap_ir": f"bootstrap_ir_internal_social_science_ir_qrels_v1_{mode}_{method_set_id}.json",
        "external_ir": f"external_ir_internal_social_science_ir_qrels_v1_{mode}_{method_set_id}.json",
        "paired_significance": f"paired_significance_internal_social_science_ir_qrels_v1_{mode}_{method_set_id}.json",
        "public_qrels_validation": "qrels_validation_public_beir_scifact_smoke.json",
    }
    artifacts = {}
    for key, filename in patterns.items():
        path = reports_dir / filename
        if path.exists():
            artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
    return artifacts


def warning_resolutions(artifacts: dict, role_by_method: dict[str, str]) -> dict:
    resolutions = {}
    stability = (artifacts.get("batch_eval") or {}).get("stability") or {}
    if stability.get("stable_top_method") is None:
        resolutions["unstable_top_method"] = {
            "action": "no_global_winner_claim",
            "rationale": "Batch-size sensitivity is retained as evidence; reports must describe a candidate set rather than a stable winner.",
            "evidence": stability,
        }
    minimum_margin = stability.get("minimum_top2_margin")
    if minimum_margin is not None and float(minimum_margin) < 0.01:
        resolutions["narrow_top2_margin"] = {
            "action": "near_tie_language_required",
            "rationale": "Top-two margins below 0.01 are treated as unresolved method differences.",
            "evidence": {"minimum_top2_margin": minimum_margin},
        }
    external_ir = artifacts.get("external_ir") or {}
    public_qrels = artifacts.get("public_qrels_validation") or {}
    if external_ir.get("benchmark_kind") == "internal_qrels":
        resolutions["internal_qrels_only"] = {
            "action": "disable_external_benchmark_claims",
            "rationale": "Internal weak qrels may exercise metrics but cannot support public benchmark claims.",
            "evidence": {
                "internal_benchmark_id": external_ir.get("benchmark_id"),
                "public_qrels_parser_status": public_qrels.get("status"),
                "public_qrels_fixture": public_qrels.get("benchmark_id"),
            },
        }
    winner_probability = (artifacts.get("bootstrap_ir") or {}).get("topline", {}).get("winner_probability")
    if winner_probability is not None and float(winner_probability) < 0.6:
        resolutions["low_bootstrap_winner_probability"] = {
            "action": "winner_probability_floor_blocks_winner_claim",
            "rationale": "The top retrieval-metric method is not accepted as identified unless bootstrap winner probability is at least 0.6.",
            "evidence": (artifacts.get("bootstrap_ir") or {}).get("topline"),
        }
    dominance = (artifacts.get("evaluation_suite") or {}).get("technical_diagnostics", {}).get("dominance") or []
    dominated_candidate_rows = [
        row for row in dominance if role_by_method.get(row.get("dominated_method_id")) != "diagnostic_baseline"
    ]
    if dominance and not dominated_candidate_rows:
        resolutions["dominated_methods"] = {
            "action": "dominated_methods_marked_diagnostic",
            "rationale": "Dominated methods are retained only as diagnostic baselines for source-quality or sensitivity analysis.",
            "evidence": dominance,
        }
    elif dominated_candidate_rows:
        resolutions["dominated_methods"] = {
            "action": "candidate_method_pruning_required",
            "rationale": "At least one dominated method is still labeled as a candidate.",
            "evidence": dominated_candidate_rows,
            "unresolved": True,
        }
    return resolutions


def global_winner_claim(artifacts: dict, resolutions: dict) -> dict:
    leaderboard = (artifacts.get("evaluation_suite") or {}).get("leaderboard") or []
    blocked_by = [
        warning_id
        for warning_id in ["unstable_top_method", "narrow_top2_margin", "low_bootstrap_winner_probability"]
        if warning_id in resolutions
    ]
    return {
        "allowed": not blocked_by,
        "blocked_by": blocked_by,
        "current_top_method": leaderboard[0]["method_id"] if leaderboard else None,
        "required_language": "candidate set / provisional ranking" if blocked_by else "stable winner",
    }


def benchmark_claim(artifacts: dict, resolutions: dict) -> dict:
    blocked = "internal_qrels_only" in resolutions
    return {
        "external_benchmark_claim_allowed": not blocked,
        "blocked_by": ["internal_qrels_only"] if blocked else [],
        "internal_qrels_allowed_for": "technical plumbing and within-corpus diagnostics",
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        f"# CANON Claim Decision: {report['mode']}",
        "",
        f"- Global winner claim allowed: {report['global_winner_claim']['allowed']}",
        f"- Required winner language: {report['global_winner_claim']['required_language']}",
        f"- External benchmark claim allowed: {report['benchmark_claim']['external_benchmark_claim_allowed']}",
        "",
        "## Resolved Warnings",
        "",
    ]
    for warning_id, resolution in report["resolutions"].items():
        lines.append(f"- `{warning_id}`: {resolution['action']} - {resolution['rationale']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CANON scientific claim-decision report.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--method-set-id", default="baseline_methods_v1")
    args = parser.parse_args()
    print(json.dumps(build_claim_decision(args.mode, args.method_set_id), indent=2))


if __name__ == "__main__":
    main()
