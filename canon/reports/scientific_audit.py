from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings


REQUIRED_ARTIFACTS = {
    "method_eval": "method_eval_{mode}_{method_set_id}.json",
    "evaluation_suite": "evaluation_suite_{mode}_{method_set_id}.json",
    "batch_eval": "batch_eval_{mode}_{method_set_id}.json",
    "probe_eval": "probe_eval_{mode}_{method_set_id}.json",
    "qrels_validation": "qrels_validation_internal_social_science_ir_qrels_v1.json",
    "external_ir": "external_ir_internal_social_science_ir_qrels_v1_{mode}_{method_set_id}.json",
    "bootstrap_ir": "bootstrap_ir_internal_social_science_ir_qrels_v1_{mode}_{method_set_id}.json",
    "paired_significance": "paired_significance_internal_social_science_ir_qrels_v1_{mode}_{method_set_id}.json",
    "faithfulness": "faithfulness_{mode}_{method_set_id}.json",
    "perturbation_eval": "perturbation_eval_{mode}_{method_set_id}.json",
    "data_card": "data_card_{mode}.json",
    "corpus_expansion": "corpus_expansion_{mode}.json",
    "public_qrels_validation": "qrels_validation_public_beir_scifact_smoke.json",
    "weight_tuning": "weight_tuning_{mode}_{method_set_id}.json",
    "claim_decision": "claim_decision_{mode}_{method_set_id}.json",
    "regression_gate": "regression_gate_{mode}_{method_set_id}.json",
    "dashboard": "dashboard_{mode}_{method_set_id}.json",
}

OPTIONAL_ARTIFACTS = {
    "diversity_audit": "diversity_audit_{mode}_{method_set_id}_diverse_k5_template_vs_lexical_k5_template.json",
    "diversity_gate": "diversity_gate_{mode}_{method_set_id}_diverse_k5_template_vs_lexical_k5_template.json",
}

RECOMMENDED_REFERENCES = [
    {
        "name": "BEIR",
        "url": "https://arxiv.org/abs/2104.08663",
        "project_relevance": "heterogeneous retrieval baselines and out-of-distribution retrieval evaluation",
    },
    {
        "name": "RAGAS",
        "url": "https://arxiv.org/abs/2309.15217",
        "project_relevance": "reference-free RAG checks for context relevance, answer relevance, and faithfulness",
    },
    {
        "name": "RAG evaluation survey",
        "url": "https://arxiv.org/abs/2405.07437",
        "project_relevance": "separate evaluation of retrieval, generation, faithfulness, robustness, and benchmarks",
    },
    {
        "name": "MTEB",
        "url": "https://arxiv.org/abs/2210.07316",
        "project_relevance": "embedding evaluation across retrieval, clustering, reranking, and other tasks",
    },
]


def build_scientific_audit(mode: str, method_set_id: str = "baseline_methods_v1") -> dict:
    settings = load_settings()
    artifacts = load_artifacts(settings.reports_dir, mode, method_set_id)
    checks = build_checks(artifacts)
    warnings = build_warnings(artifacts)
    audit = {
        "mode": mode,
        "method_set_id": method_set_id,
        "status": audit_status(checks),
        "checks": checks,
        "warnings": warnings,
        "resolved_warnings": resolved_warnings(artifacts),
        "topline": topline(artifacts),
        "recommended_external_references": RECOMMENDED_REFERENCES,
        "next_scientific_steps": next_scientific_steps(),
    }
    stem = f"scientific_audit_{mode}_{method_set_id}"
    write_json(settings.reports_dir / f"{stem}.json", audit)
    (settings.reports_dir / f"{stem}.md").write_text(render_markdown(audit), encoding="utf-8")
    return audit


def load_artifacts(reports_dir: Path, mode: str, method_set_id: str) -> dict:
    artifacts = {}
    for key, pattern in REQUIRED_ARTIFACTS.items():
        path = reports_dir / pattern.format(mode=mode, method_set_id=method_set_id)
        if path.exists():
            artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
    for key, pattern in OPTIONAL_ARTIFACTS.items():
        path = reports_dir / pattern.format(mode=mode, method_set_id=method_set_id)
        if path.exists():
            artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = reports_dir / f"experiment_manifest_{mode}_full.json"
    if manifest_path.exists():
        artifacts["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return artifacts


def build_checks(artifacts: dict) -> list[dict]:
    checks = []
    for key in REQUIRED_ARTIFACTS:
        checks.append(check(f"artifact:{key}", key in artifacts, f"Required artifact '{key}' exists."))
    suite = artifacts.get("evaluation_suite", {})
    probe = artifacts.get("probe_eval", {})
    batches = artifacts.get("batch_eval", {})
    tuning = artifacts.get("weight_tuning", {})
    faithfulness = artifacts.get("faithfulness", {})
    perturbations = artifacts.get("perturbation_eval", {})
    regression_gate = artifacts.get("regression_gate", {})
    qrels_validation = artifacts.get("qrels_validation", {})
    data_card = artifacts.get("data_card", {})
    corpus_expansion = artifacts.get("corpus_expansion", {})
    claim_decision = artifacts.get("claim_decision", {})
    diversity_gate = artifacts.get("diversity_gate", {})
    checks.extend(
        [
            check("query_count>=25", int(suite.get("query_count") or 0) >= 25, "Main eval has at least 25 technical queries."),
            check("method_count>=5", len(suite.get("leaderboard") or []) >= 5, "Main eval compares at least five retrieval methods."),
            check(
                "probe_failures==0",
                int(probe.get("failure_summary", {}).get("failure_count") or 0) == 0,
                "Adversarial technical probe suite has zero expectation failures.",
            ),
            check(
                "batch_count>=4",
                len(batches.get("batches") or []) >= 4,
                "Batch stability was measured at four or more query-set sizes.",
            ),
            check(
                "external_ir_query_count>=5",
                int((artifacts.get("external_ir") or {}).get("query_count") or 0) >= 5,
                "BEIR/TREC-style retrieval metrics were exercised on at least five qrels queries.",
            ),
            check(
                "qrels_validation==pass",
                qrels_validation.get("status") == "pass",
                "Qrels validation passes before retrieval benchmark metrics are reported.",
            ),
            check(
                "weight_variants>=5",
                int(tuning.get("variant_count") or 0) >= 5,
                "Metric sensitivity was tested across at least five weighting variants.",
            ),
            check(
                "faithfulness_query_count>=5",
                int(faithfulness.get("query_count") or 0) >= 5,
                "Citation faithfulness was measured across at least five synthesis queries.",
            ),
            check(
                "perturbation_query_count>=8",
                int(perturbations.get("query_count") or 0) >= 8,
                "Query perturbation robustness was measured across at least eight queries.",
            ),
            check(
                "regression_gate==pass",
                regression_gate.get("status") == "pass",
                "Regression gates pass for the generated evaluation bundle.",
            ),
            check(
                "data_card_work_count>=25",
                int(data_card.get("work_count") or 0) >= 25,
                "Corpus data card describes at least 25 works.",
            ),
            check(
                "corpus_expansion_profile_present",
                bool(corpus_expansion.get("profile_id")),
                "Corpus expansion profile and source registry are present.",
            ),
            check(
                "claim_decision==pass",
                claim_decision.get("status") == "pass",
                "Scientific claim-decision report exists and passes.",
            ),
        ]
    )
    if diversity_gate:
        checks.append(
            check(
                "diversity_gate==pass",
                diversity_gate.get("status") == "pass",
                "Diversity-first regression gates pass for useful breadth, off-topic risk, and noise.",
            )
        )
    return checks


def build_warnings(artifacts: dict) -> list[dict]:
    warnings = []
    batches = artifacts.get("batch_eval", {})
    stability = batches.get("stability") or {}
    suite = artifacts.get("evaluation_suite", {})
    external_ir = artifacts.get("external_ir", {})
    bootstrap_ir = artifacts.get("bootstrap_ir", {})
    data_card = artifacts.get("data_card", {})
    corpus_expansion = artifacts.get("corpus_expansion", {})
    resolved_ids = set((artifacts.get("claim_decision") or {}).get("resolved_warning_ids") or [])
    if stability.get("stable_top_method") is None and "unstable_top_method" not in resolved_ids:
        warnings.append(
            {
                "id": "unstable_top_method",
                "message": "Top method changes across batch sizes; report winner as provisional and near-tied.",
                "evidence": stability,
            }
        )
    minimum_margin = stability.get("minimum_top2_margin")
    if (
        minimum_margin is not None
        and float(minimum_margin) < 0.01
        and "narrow_top2_margin" not in resolved_ids
    ):
        warnings.append(
            {
                "id": "narrow_top2_margin",
                "message": "Top-two margin is narrow; use stress slices and external benchmarks before strong claims.",
                "evidence": {"minimum_top2_margin": minimum_margin},
            }
        )
    if external_ir.get("benchmark_kind") == "internal_qrels" and "internal_qrels_only" not in resolved_ids:
        warnings.append(
            {
                "id": "internal_qrels_only",
                "message": "External-IR machinery currently uses internal weak qrels; add BEIR/TREC qrels before benchmark claims.",
                "evidence": {
                    "benchmark_id": external_ir.get("benchmark_id"),
                    "query_count": external_ir.get("query_count"),
                },
            }
        )
    winner_probability = bootstrap_ir.get("topline", {}).get("winner_probability")
    if (
        winner_probability is not None
        and float(winner_probability) < 0.6
        and "low_bootstrap_winner_probability" not in resolved_ids
    ):
        warnings.append(
            {
                "id": "low_bootstrap_winner_probability",
                "message": "Bootstrap resampling does not strongly identify a retrieval-metric winner.",
                "evidence": bootstrap_ir.get("topline"),
            }
        )
    if data_card.get("limitations"):
        warnings.append(
            {
                "id": "data_card_limitations",
                "message": "Corpus data card lists limitations that constrain scientific claims.",
                "evidence": data_card.get("limitations"),
            }
        )
    if (corpus_expansion.get("readiness") or {}).get("status") == "needs_expansion":
        warnings.append(
            {
                "id": "corpus_expansion_needed",
                "message": "Current corpus has not reached the product expansion target.",
                "evidence": corpus_expansion.get("readiness"),
            }
        )
    dominance = suite.get("technical_diagnostics", {}).get("dominance") or []
    if dominance and "dominated_methods" not in resolved_ids:
        warnings.append(
            {
                "id": "dominated_methods",
                "message": "At least one method is dominated on the current metric set.",
                "evidence": dominance,
            }
        )
    diversity_gate = artifacts.get("diversity_gate", {})
    if diversity_gate and diversity_gate.get("status") != "pass":
        warnings.append(
            {
                "id": "diversity_gate_failed",
                "message": "Diversity-first regression gate failed; do not present diversity-first as product-ready.",
                "evidence": diversity_gate.get("checks"),
            }
        )
    return warnings


def resolved_warnings(artifacts: dict) -> list[dict]:
    decision = artifacts.get("claim_decision") or {}
    resolutions = decision.get("resolutions") or {}
    return [
        {
            "id": warning_id,
            "action": resolution.get("action"),
            "rationale": resolution.get("rationale"),
        }
        for warning_id, resolution in sorted(resolutions.items())
        if not resolution.get("unresolved")
    ]


def topline(artifacts: dict) -> dict:
    suite = artifacts.get("evaluation_suite", {})
    probe = artifacts.get("probe_eval", {})
    batches = artifacts.get("batch_eval", {})
    diversity_audit = artifacts.get("diversity_audit", {})
    diversity_gate = artifacts.get("diversity_gate", {})
    leaderboard = suite.get("leaderboard") or []
    probe_leaderboard = probe.get("probe_summary", {}).get("leaderboard") or []
    return {
        "main_eval_winner": leaderboard[0] if leaderboard else None,
        "probe_winner": probe_leaderboard[0] if probe_leaderboard else None,
        "probe_failure_count": probe.get("failure_summary", {}).get("failure_count"),
        "batch_stability": batches.get("stability"),
        "diversity_audit": diversity_audit.get("aggregate"),
        "diversity_gate_status": diversity_gate.get("status"),
    }


def next_scientific_steps() -> list[str]:
    return [
        "Replace or supplement the internal weak qrels with public BEIR/TREC-style benchmark qrels.",
        "Supplement lexical faithfulness checks with an external NLI or judge-based verifier.",
        "Freeze a preregistered experiment protocol before tuning another method.",
        "Run larger-query bootstrap and paired significance checks over public benchmark qrels.",
        "Expand data-card provenance with explicit inclusion/exclusion criteria and coverage comparisons.",
    ]


def check(identifier: str, passed: bool, message: str) -> dict:
    return {"id": identifier, "passed": bool(passed), "message": message}


def audit_status(checks: list[dict]) -> str:
    return "pass" if all(item["passed"] for item in checks) else "fail"


def render_markdown(audit: dict) -> str:
    lines = [
        f"# CANON Scientific Audit: {audit['mode']}",
        "",
        f"Status: **{audit['status']}**",
        "",
        "## Checks",
        "",
    ]
    for item in audit["checks"]:
        marker = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- {marker}: `{item['id']}` - {item['message']}")
    lines.extend(["", "## Warnings", ""])
    if audit["warnings"]:
        for item in audit["warnings"]:
            lines.append(f"- WARN: `{item['id']}` - {item['message']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Resolved Warnings", ""])
    if audit.get("resolved_warnings"):
        for item in audit["resolved_warnings"]:
            lines.append(f"- RESOLVED: `{item['id']}` - {item['action']}: {item['rationale']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Topline", "", "```json", json.dumps(audit["topline"], indent=2), "```", ""])
    lines.extend(["## External Evaluation Anchors", ""])
    for reference in audit["recommended_external_references"]:
        lines.append(f"- [{reference['name']}]({reference['url']}): {reference['project_relevance']}")
    lines.extend(["", "## Next Scientific Steps", ""])
    for step in audit["next_scientific_steps"]:
        lines.append(f"- {step}")
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a scientific defensibility audit for CANON reports.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--method-set-id", default="baseline_methods_v1")
    args = parser.parse_args()
    print(json.dumps(build_scientific_audit(args.mode, args.method_set_id), indent=2))


if __name__ == "__main__":
    main()
